from utils import *  # including 'init_distributed', 'weight_loader'
from trainer import Trainer
import os
import sys
import torch
from pathlib import Path


def get_arguments(base_path):
    parser = argparse.ArgumentParser(description='Multi-Band Brain Net (MBBN)')

    # ── Experiment ──────────────────────────────────────────────────────────
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--dataset_name', type=str,
                        choices=['ABCD', 'ABIDE', 'UKB'], default='ABCD')
    parser.add_argument('--step', default='2',
                        choices=['1', '2', '3', '4'],
                        help='1=vanilla_BERT  2=MBBN  3=pretraining  4=test')
    parser.add_argument('--target', type=str, default='sex')
    parser.add_argument('--fine_tune_task',
                        choices=['regression', 'binary_classification'])
    parser.add_argument('--seed', type=int, default=1)

    # ── Data paths ──────────────────────────────────────────────────────────
    parser.add_argument('--abcd_path',
                        default='/storage/bigdata/ABCD/ABCD_ROI/7.ROI')
    parser.add_argument('--ukb_path',
                        default='/storage/bigdata/UKB/fMRI/UKB_ROI')
    parser.add_argument('--abide_path',
                        default='/scratch/connectome/stellasybae/ABIDE_ROI')
    parser.add_argument('--base_path', default=base_path)
    parser.add_argument('--log_dir',
                        default=os.path.join(base_path, 'runs'))

    # ── Model architecture ──────────────────────────────────────────────────
    # intermediate_vec = number of ROIs (360 for HCP-MMP1, 400 for Schaefer)
    parser.add_argument('--intermediate_vec', type=int)
    # num_heads must divide intermediate_vec  (12 for HCP-MMP1, 8 for Schaefer)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--transformer_hidden_layers', type=int, default=8)
    parser.add_argument('--transformer_dropout_rate', type=float, default=0.3)

    # ── Frequency decomposition ─────────────────────────────────────────────
    # filtering_type: Boxcar (default) works well for most datasets;
    #   use FIR for ABCD + HCP-MMP1 (see Table 7 in paper)
    parser.add_argument('--filtering_type', default='Boxcar',
                        choices=['FIR', 'Boxcar'])

    # ── Spatial loss (λ) ────────────────────────────────────────────────────
    # Optimal values per dataset+atlas (Table 7): ABCD→10, UKB→1, ABIDE→100
    parser.add_argument('--spatial_loss_factor', type=float, default=1.0)

    # ── Transfer learning ───────────────────────────────────────────────────
    parser.add_argument('--pretrained_model_weights_path', default=None)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_test', action='store_true',
                        help='run test phase of a finetuned model')

    # ── Distributed / optimization ──────────────────────────────────────────
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world_size', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--init_method', default='file', type=str,
                        choices=['file', 'env'])
    parser.add_argument('--amp', action='store_false',
                        help='pass to DISABLE automatic mixed precision')
    parser.add_argument('--gradient_clipping', action='store_true')
    parser.add_argument('--clip_max_norm', type=float, default=1.0)
    parser.add_argument('--accumulation_steps', default=1, type=int)

    # ── W&B logging ─────────────────────────────────────────────────────────
    parser.add_argument('--wandb_mode', default='online',
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', default='', type=str)
    parser.add_argument('--wandb_project', default='MBBN', type=str)

    # ── Analysis tools ──────────────────────────────────────────────────────
    parser.add_argument('--flop_counter', action='store_true')
    parser.add_argument('--weightwatcher', action='store_true')
    parser.add_argument('--weightwatcher_save_dir', default=None)

    # ── Phase 1: vanilla BERT (baseline) ────────────────────────────────────
    parser.add_argument('--task_phase1', type=str, default='vanilla_BERT')
    parser.add_argument('--batch_size_phase1', type=int, default=8)
    parser.add_argument('--nEpochs_phase1', type=int, default=100)
    parser.add_argument('--optim_phase1', default='AdamW')
    parser.add_argument('--weight_decay_phase1', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase1', default='SGDR')
    parser.add_argument('--lr_init_phase1', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase1', type=float, default=0.97)
    parser.add_argument('--lr_step_phase1', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase1', type=int, default=500)
    # sequence_length: UKB=464, ABCD=348, ABIDE=280
    parser.add_argument('--sequence_length_phase1', type=int, default=348)
    parser.add_argument('--workers_phase1', type=int, default=4)

    # ── Phase 2: MBBN (from scratch or finetune) ─────────────────────────────
    parser.add_argument('--task_phase2', type=str, default='MBBN')
    parser.add_argument('--batch_size_phase2', type=int, default=8)
    parser.add_argument('--nEpochs_phase2', type=int, default=100)
    parser.add_argument('--optim_phase2', default='AdamW')
    parser.add_argument('--weight_decay_phase2', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase2', default='SGDR')
    parser.add_argument('--lr_init_phase2', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase2', type=float, default=0.97)
    parser.add_argument('--lr_step_phase2', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase2', type=int, default=500)
    parser.add_argument('--sequence_length_phase2', type=int, default=348)
    parser.add_argument('--workers_phase2', type=int, default=4)

    # ── Phase 3: pretraining ─────────────────────────────────────────────────
    parser.add_argument('--task_phase3', type=str, default='MBBN_pretraining')
    parser.add_argument('--batch_size_phase3', type=int, default=4)
    parser.add_argument('--nEpochs_phase3', type=int, default=400)
    parser.add_argument('--optim_phase3', default='AdamW')
    parser.add_argument('--weight_decay_phase3', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase3', default='SGDR')
    parser.add_argument('--lr_init_phase3', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase3', type=float, default=0.97)
    parser.add_argument('--lr_step_phase3', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase3', type=int, default=500)
    # NOTE: this used to be `--sequence_length_phase4` inside the phase-3
    # block, so sort_args('3', ...) discarded it and pretraining silently ran at
    # the phase-3 default. pretrain_MBBN.slurm passed 464 and got 348.
    parser.add_argument('--sequence_length_phase3', type=int, default=464)
    parser.add_argument('--workers_phase3', type=int, default=4)
    # num_hub_ROIs: top-k high-communicability nodes to mask (default 380 for Schaefer 400)
    parser.add_argument('--num_hub_ROIs', type=int, default=380)

    # ── Phase 4: test ────────────────────────────────────────────────────────
    parser.add_argument('--task_phase4', type=str, default='test')
    parser.add_argument('--model_weights_path_phase4', default=None)
    parser.add_argument('--batch_size_phase4', type=int, default=4)
    parser.add_argument('--nEpochs_phase4', type=int, default=1)
    parser.add_argument('--optim_phase4', default='AdamW')
    parser.add_argument('--weight_decay_phase4', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase4', default='SGDR')
    parser.add_argument('--lr_init_phase4', type=float, default=1e-4)
    parser.add_argument('--lr_gamma_phase4', type=float, default=0.9)
    parser.add_argument('--lr_step_phase4', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase4', type=int, default=100)
    parser.add_argument('--sequence_length_phase4', type=int, default=464)
    parser.add_argument('--workers_phase4', type=int, default=4)

    # ── Hardening options (see docs/audit/AUDIT.md) ─────────────────────────
    # Every default below reproduces the published behaviour EXCEPT where the
    # published behaviour is a defect; those are marked (changed).
    hard = parser.add_argument_group('hardening')
    hard.add_argument('--head_type', choices=['published', 'linear', 'mlp'], default='linear',
                      help='(changed) prediction head. "published" = Linear->BatchNorm1d(1)'
                           '->Dropout(0.6) on the final logit, which makes a subject\'s '
                           'prediction depend on its batch-mates and zeroes 61%% of training '
                           'logits. "linear" is a plain read-out.')
    hard.add_argument('--head_dropout', type=float, default=0.6)
    hard.add_argument('--spat_diff_loss_type', default='minus_log',
                      choices=['minus_log', 'minus_log_eps', 'neg_linear', 'cosine'],
                      help='band-repulsion form. "minus_log" is published: unbounded, '
                           'singular where the three maps agree, and its global optimum is '
                           'one-hot (hub) attention rows.')
    hard.add_argument('--spat_diff_eps', type=float, default=1e-4)
    hard.add_argument('--spat_diff_entropy_weight', type=float, default=0.0,
                      help='penalise low-entropy attention rows, counteracting hub collapse')
    hard.add_argument('--spatial_loss_warmup', type=int, default=0,
                      help='linearly ramp spatial_loss_factor over this many optimizer steps')
    hard.add_argument('--band_embedding', action='store_true',
                      help='learned additive band identity for the shared temporal encoder')
    hard.add_argument('--spatial_head', action='store_true',
                      help='put the band-specific spatial attention maps on the prediction '
                           'path. Without this d(prediction)/d(spatial attention) is exactly '
                           '0, so those maps carry no label gradient and cannot be attributed.')
    hard.add_argument('--attn_only', action='store_true',
                      help='drop the unused value projection in the spatial attention')
    hard.add_argument('--use_padding_mask', action='store_true', default=True,
                      help='(changed) exclude zero-padded timepoints from attention')
    hard.add_argument('--no_padding_mask', dest='use_padding_mask', action='store_false')
    hard.add_argument('--pretrained_sequence_length', type=int, default=None,
                      help='sequence length the pretrained checkpoint was trained at; '
                           'replaces the hard-coded 464')
    # masked pretraining
    hard.add_argument('--random_mask', action='store_true',
                      help='resample the spatial/temporal mask per subject per epoch. The '
                           'published mask is identical for every subject and every epoch.')
    hard.add_argument('--mask_ratio_spatial', type=float, default=None,
                      help='fraction of ROIs to hide (default: num_hub_ROIs/intermediate_vec, '
                           'i.e. 380/400 = 95%% as released)')
    hard.add_argument('--mask_ratio_temporal', type=float, default=None)
    hard.add_argument('--mask_loss_on_masked_only', action='store_true',
                      help='score reconstruction only where the input was hidden')
    hard.add_argument('--communicability_dir', default=None)
    hard.add_argument('--communicability_dataset', default='UKB')
    # data pipeline
    hard.add_argument('--cache_bands', action='store_true',
                      help='memoise the per-subject knee fit + band filtering to disk; the '
                           'released code repeats it every epoch')
    hard.add_argument('--band_cache_dir', default=None)
    # splitting / evaluation protocol
    hard.add_argument('--split_seed', type=int, default=None,
                      help='seed for the train/val/test partition; defaults to --seed')
    hard.add_argument('--site_stratify', action='store_true',
                      help='stratify the split on target x acquisition site')
    hard.add_argument('--group_by_family', action='store_true',
                      help='keep siblings/twins (ABCD) within a single fold')
    hard.add_argument('--leave_one_site_out', default=None,
                      help='assign this site entirely to the test fold')
    hard.add_argument('--eval_test_every_epoch', action='store_true',
                      help='score the test set every epoch (published behaviour). Off by '
                           'default: selection uses validation only.')
    hard.add_argument('--skip_final_test', action='store_true',
                      help='do not evaluate the selected checkpoint on the test fold')
    hard.add_argument('--pos_weight', type=float, default=None,
                      help='BCEWithLogitsLoss pos_weight for imbalanced targets')
    hard.add_argument('--nan_policy', choices=['warn', 'raise'], default='warn',
                      help='(published: warn) "raise" stops instead of substituting 0 for a '
                           'NaN loss and continuing on a corrupted objective')
    hard.add_argument('--deterministic', action='store_true')
    hard.add_argument('--compile_model', action='store_true')
    hard.add_argument('--keep_all_best_checkpoints', action='store_true',
                      help='published behaviour: one file per improving epoch (~325 MB each)')

    args = parser.parse_args()
    # remember what the user actually typed, so sort_args can warn about options
    # that belong to a different phase and would be silently discarded
    args._explicit = {a.lstrip('-').replace('-', '_').split('=')[0]
                      for a in sys.argv[1:] if a.startswith('--')}

    # Standard MBBN settings — not user-adjustable
    args.fmri_type = 'divided_timeseries'
    args.fmri_dividing_type = 'three_channels'
    args.dividing_method = 'lorentzian'
    args.use_raw_knee = True
    args.seq_part = 'head'
    args.use_high_freq = True
    args.spatiotemporal = True
    args.attn_mask = True
    # Pretraining masking settings (step 3)
    args.use_mask_loss = True
    args.masking_method = 'spatiotemporal'
    args.spatial_masking_type = 'hub_ROIs'
    args.communicability_option = 'remove_high_comm_node'
    args.temporal_masking_type = 'time_window'
    args.temporal_masking_window_size = 20
    args.window_interval_rate = 2
    args.cuda = torch.cuda.is_available()

    return args


def setup_folders(base_path):
    os.makedirs(os.path.join(base_path, 'experiments'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'splits'), exist_ok=True)


def run_phase(args, loaded_model_weights_path, phase_num, phase_name):
    experiment_folder = '{}_{}_{}_{}'.format(
        args.dataset_name, phase_name, args.target, args.exp_name)
    experiment_folder = Path(os.path.join(
        args.base_path, 'experiments', experiment_folder))
    os.makedirs(experiment_folder, exist_ok=True)

    setattr(args, 'loaded_model_weights_path_phase' + phase_num,
            loaded_model_weights_path)
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name

    print(f'Saving results to {args.experiment_folder}')
    args_logger(args)

    kwargs = sort_args(phase_num, vars(args), explicit=getattr(args, '_explicit', None))
    reproducibility(**kwargs)

    trainer = Trainer(sets=['train', 'val', 'test'], **kwargs)
    trainer.training()

    # The released code assembled '<title>_BEST_val_loss.pth' (or '..._accuracy'),
    # but save_checkpoint_ writes '<title>_epoch_<n>_BEST_val_AUROC|MAE|loss.pth',
    # so the returned path never existed. Report what was actually written.
    model_weights_path = trainer.best_checkpoint_path
    if model_weights_path is None:
        model_weights_path = os.path.join(
            trainer.writer.experiment_folder,
            trainer.writer.experiment_title + '_last_epoch.pth')
        print('validation never improved; falling back to the last-epoch checkpoint')
    print(f'selected checkpoint: {model_weights_path}')
    return model_weights_path


def test(args, phase_num, model_weights_path):
    experiment_folder = '{}_{}_{}'.format(
        args.dataset_name, 'test_{}'.format(args.fine_tune_task), args.exp_name)
    experiment_folder = Path(os.path.join(
        args.base_path, 'tests', experiment_folder))
    os.makedirs(experiment_folder, exist_ok=True)

    setattr(args, 'loaded_model_weights_path_phase' + phase_num,
            model_weights_path)
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    args_logger(args)

    kwargs = sort_args(args.step, vars(args), explicit=getattr(args, '_explicit', None))
    reproducibility(**kwargs)
    trainer = Trainer(sets=['test'], **kwargs)
    trainer.testing()
    return trainer


if __name__ == '__main__':
    args = get_arguments(os.getcwd())
    # setup_folders() used to run against the cwd before parsing, so --base_path
    # only half-applied: splits/experiments were read from base_path but the
    # directories were created next to wherever the job happened to start.
    setup_folders(args.base_path)

    init_distributed(args)
    model_weights_path, step, task = weight_loader(args)

    if step == '4':
        test(args, '4', model_weights_path)
    else:
        print(f'Starting phase {step}: {task}')
        best = run_phase(args, model_weights_path, step, task)
        print(f'Finished phase {step}: {task}')
        # Select on validation, then score the held-out fold exactly once with
        # the validation operating point. Scoring test every epoch (published
        # behaviour) is available via --eval_test_every_epoch.
        if step in ('1', '2') and not args.skip_final_test and best and os.path.exists(best):
            print('Evaluating the selected checkpoint on the held-out test fold')
            test(args, step, best)
