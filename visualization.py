"""Attribution over MBBN's band-specific spatial attention maps.

What changed and why
--------------------
The released script computed its saved "gradient" by backpropagating the
*band-repulsion loss*::

    spat_diff_loss = -torch.log(L1(h,l) + L1(h,u) + L1(l,u))
    spat_diff_loss.backward()

Two measured consequences (``docs/audit/AUDIT.md``):

1. That gradient is analytically degenerate. With ``S`` a sum of mean-L1
   distances, ``d(-log S)/dh_ij = -[sign(h-l)+sign(h-u)] / (numel * S)``, so it
   takes only the five values ``{-2,-1,0,1,2}`` times one global scalar. Measured
   on a (2,8,400,400) map: 4 distinct values over 2.56M entries, 33% exactly
   zero. Gradient *magnitude* therefore carries no per-edge information.
2. It contains no label information. The diagnosis enters only through which
   subjects are selected for averaging, and in the released model
   ``d(prediction)/d(spatial attention) == 0`` exactly, because the spatial
   branch is not on the prediction path at all.

``--attribution label_gradient`` (default) attributes the *prediction* instead,
which requires the maps to be connected to it -- train with ``--spatial_head``.
``--attribution spatial_difference`` reproduces the published behaviour.
``--n_permutations`` adds a label-permutation null so a group difference in
these maps can be compared against chance rather than reported raw.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from model import *
from trainer import *
from data_preprocess_and_load.dataloaders import *


def get_arguments(base_path=os.getcwd()):
    parser = argparse.ArgumentParser(description='MBBN interpretability')

    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--dataset_name', type=str, choices=['ABCD', 'ABIDE', 'UKB'], default='ABCD')
    parser.add_argument('--target', type=str, default='sex')
    parser.add_argument('--fine_tune_task', choices=['regression', 'binary_classification'],
                        default='binary_classification')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split_seed', type=int, default=None)

    parser.add_argument('--abcd_path', default='/storage/bigdata/ABCD/ABCD_ROI/7.ROI')
    parser.add_argument('--ukb_path', default='/storage/bigdata/UKB/fMRI/UKB_ROI')
    parser.add_argument('--abide_path', default='/scratch/connectome/stellasybae/ABIDE_ROI')
    parser.add_argument('--base_path', default=base_path)
    parser.add_argument('--log_dir', type=str, default=os.path.join(base_path, 'runs'))

    parser.add_argument('--intermediate_vec', type=int, default=360)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--transformer_hidden_layers', type=int, default=8)
    parser.add_argument('--transformer_dropout_rate', type=float, default=0.3)
    parser.add_argument('--filtering_type', default='Boxcar', choices=['FIR', 'Boxcar'])
    parser.add_argument('--spatial_loss_factor', type=float, default=1.0)

    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world_size', default=-1, type=int)
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--wandb_mode', default='disabled', choices=['online', 'offline', 'disabled'])

    parser.add_argument('--task', type=str, default='test')
    parser.add_argument('--step', default='2', choices=['1', '2', '3', '4'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=1)
    parser.add_argument('--optim', default='AdamW')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lr_policy', default='SGDR')
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=1500)
    parser.add_argument('--lr_warmup', type=int, default=100)
    parser.add_argument('--sequence_length', type=int, default=348)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    # ── attribution ─────────────────────────────────────────────────────────
    parser.add_argument('--attribution', choices=['label_gradient', 'spatial_difference'],
                        default='label_gradient',
                        help='what to backpropagate. "spatial_difference" is the published '
                             'choice and is label-independent.')
    parser.add_argument('--confidence_margin', type=float, default=0.0,
                        help='keep a subject only if |p - 0.5| >= this. The released script '
                             'used 0, i.e. "correctly classified" rather than "confident".')
    parser.add_argument('--n_permutations', type=int, default=0,
                        help='label permutations for the group-difference null (0 = skip)')
    parser.add_argument('--head_type', choices=['published', 'linear', 'mlp'], default='linear')
    parser.add_argument('--head_dropout', type=float, default=0.6)
    parser.add_argument('--band_embedding', action='store_true')
    parser.add_argument('--spatial_head', action='store_true')
    parser.add_argument('--attn_only', action='store_true')
    parser.add_argument('--use_padding_mask', action='store_true', default=True)
    parser.add_argument('--cache_bands', action='store_true')
    parser.add_argument('--band_cache_dir', default=None)
    parser.add_argument('--pretrained_sequence_length', type=int, default=None)
    parser.add_argument('--site_stratify', action='store_true')
    parser.add_argument('--group_by_family', action='store_true')
    parser.add_argument('--leave_one_site_out', default=None)

    args = parser.parse_args()

    args.fmri_type = 'divided_timeseries'
    args.fmri_dividing_type = 'three_channels'
    args.dividing_method = 'lorentzian'
    args.use_raw_knee = True
    args.seq_part = 'head'
    args.use_high_freq = True
    args.spatiotemporal = True
    args.spat_diff_loss_type = 'minus_log'
    args.attn_mask = True
    args.cuda = torch.cuda.is_available()
    args.visualization = False
    args.finetune = False
    args.finetune_test = False
    args.pretrained_model_weights_path = None
    return args


BANDS = ('high', 'low', 'ultralow')


def build_hooks(model):
    """Register forward/backward hooks once and return (store, handles).

    The released script re-registered hooks *inside* the per-subject loop and
    never removed them, so by subject k every module carried k forward and k
    backward hooks, each writing into a different dict that stayed alive through
    its closure. It also used `register_backward_hook`, which torch documents as
    returning incomplete grad_input for modules with several autograd nodes, and
    stored `output[0]` for both hooks -- meaning the forward payload was subject
    0's map while the backward payload was the whole batch's gradient.
    """
    store = {'act': {}, 'grad': {}}
    handles = []
    for band in BANDS:
        mod = getattr(model, f'{band}_spatial_attention')

        def fwd(_m, _i, out, band=band):
            store['act'][band] = out.detach()

        def bwd(_m, _gi, go, band=band):
            store['grad'][band] = go[0].detach()

        handles.append(mod.register_forward_hook(fwd))
        handles.append(mod.register_full_backward_hook(bwd))
    return store, handles


def permutation_null(maps, labels, n_perm, rng):
    """Label-permutation null for a group difference in attribution maps.

    `maps`: (n_subjects, N, N). Returns the observed difference map, the
    max-statistic null distribution, and the family-wise-corrected p-value map.
    """
    labels = np.asarray(labels)
    a, b = labels == 1, labels == 0
    if a.sum() < 2 or b.sum() < 2:
        return None
    obs = maps[a].mean(0) - maps[b].mean(0)
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        d = maps[perm == 1].mean(0) - maps[perm == 0].mean(0)
        null_max[i] = np.abs(d).max()
    p_fwe = (null_max[None, None, :] >= np.abs(obs)[:, :, None]).mean(-1)
    return obs, null_max, p_fwe


def main():
    args = get_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if args.cuda else 'cpu')

    model = Transformer_Finetune_Three_Channels(**vars(args))
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys')
        print(f'  missing[:5]    = {list(missing)[:5]}')
        print(f'  unexpected[:5] = {list(unexpected)[:5]}')
    model.eval().to(device)

    if args.attribution == 'label_gradient' and not args.spatial_head:
        sys.exit(
            'ERROR: --attribution label_gradient needs the spatial attention maps to be on '
            'the prediction path, but this model was built without --spatial_head, so '
            'd(prediction)/d(spatial attention) is exactly 0 and every attribution would be '
            'zero. Either retrain with --spatial_head, or pass '
            '--attribution spatial_difference to reproduce the published (label-independent) '
            'maps.')

    data_handler = DataHandler(**vars(args))
    loaders = data_handler.create_dataloaders()
    test_loader = loaders[2]

    store, handles = build_hooks(model)
    l1 = nn.L1Loss()
    records, per_band = [], {b: [] for b in BANDS}

    try:
        for data in tqdm(test_loader):
            subj_name = data['subject_name'][0]
            xs = [data[f'fmri_{k}freq_sequence'].float().to(device)
                  for k in ('high', 'low', 'ultralow')]
            vm = data.get('valid_mask')
            vm = vm.to(device) if vm is not None else None
            label = float(data[args.target].reshape(-1)[0])

            model.zero_grad(set_to_none=True)
            out = model(*xs, valid_mask=vm)
            logit = out[args.fine_tune_task].reshape(-1)[0]
            p = torch.sigmoid(logit).item()
            pred_int = int(p > 0.5)

            if pred_int != int(label) or abs(p - 0.5) < args.confidence_margin:
                continue

            if args.attribution == 'label_gradient':
                # gradient of the model's own decision variable
                logit.backward()
            else:
                # published: gradient of the band-repulsion loss. Independent of
                # the label, and quantised to a handful of values.
                h, l_, u = (out[f'{b}_spatial_attention'] for b in BANDS)
                (-torch.log(l1(h, l_) + l1(h, u) + l1(l_, u))).backward()

            rec = {'subject': str(subj_name), 'label': label, 'prob': p}
            for band in BANDS:
                act = store['act'][band].mean(dim=1)[0].float().cpu().numpy()   # heads-averaged
                grd = store['grad'][band].mean(dim=1)[0].float().cpu().numpy()
                per_band[band].append(act * grd)
                rec[f'{band}_grad_unique'] = int(np.unique(grd).size)
            records.append(rec)
            np.savez_compressed(
                os.path.join(args.save_dir, f'{subj_name}_attribution.npz'),
                **{f'{b}_act': store['act'][b].mean(dim=1)[0].float().cpu().numpy() for b in BANDS},
                **{f'{b}_grad': store['grad'][b].mean(dim=1)[0].float().cpu().numpy() for b in BANDS},
                label=label, prob=p)
    finally:
        for h in handles:
            h.remove()

    summary = {'attribution': args.attribution, 'n_subjects': len(records),
               'confidence_margin': args.confidence_margin}
    if records:
        summary['median_distinct_gradient_values'] = {
            b: float(np.median([r[f'{b}_grad_unique'] for r in records])) for b in BANDS}

    if args.n_permutations and len(records) >= 8:
        rng = np.random.default_rng(args.seed)
        labels = np.array([r['label'] for r in records])
        summary['permutation'] = {}
        for band in BANDS:
            res = permutation_null(np.stack(per_band[band]), labels, args.n_permutations, rng)
            if res is None:
                continue
            obs, null_max, p_fwe = res
            np.savez_compressed(os.path.join(args.save_dir, f'group_null_{band}.npz'),
                                observed=obs, null_max=null_max, p_fwe=p_fwe)
            summary['permutation'][band] = {
                'n_edges_p_fwe_lt_0.05': int((p_fwe < 0.05).sum()),
                'max_abs_observed': float(np.abs(obs).max()),
                'null_max_p95': float(np.percentile(null_max, 95))}

    with open(os.path.join(args.save_dir, 'attribution_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
