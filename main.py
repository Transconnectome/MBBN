from utils import *  # including 'init_distributed', 'weight_loader'
from trainer import Trainer
import os
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
                        choices=['online', 'offline'])
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
    parser.add_argument('--sequence_length_phase4', type=int, default=464)
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
    parser.add_argument('--sequence_length_phase3', type=int, default=348)
    parser.add_argument('--workers_phase4', type=int, default=4)

    args = parser.parse_args()

    # Standard MBBN settings — not user-adjustable
    args.fmri_type = 'divided_timeseries'
    args.fmri_dividing_type = 'three_channels'
    args.dividing_method = 'lorentzian'
    args.use_raw_knee = True
    args.seq_part = 'head'
    args.use_high_freq = True
    args.spatiotemporal = True
    args.spat_diff_loss_type = 'minus_log'
    args.attn_mask = True
    # Pretraining masking settings (step 3)
    args.use_mask_loss = True
    args.masking_method = 'spatiotemporal'
    args.spatial_masking_type = 'hub_ROIs'
    args.communicability_option = 'remove_high_comm_node'
    args.temporal_masking_type = 'time_window'
    args.temporal_masking_window_size = 20
    args.window_interval_rate = 2
    args.cuda = True

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

    kwargs = sort_args(phase_num, vars(args))
    reproducibility(**kwargs)

    trainer = Trainer(sets=['train', 'val', 'test'], **kwargs)
    trainer.training()

    if phase_num == '3' and not args.fine_tune_task == 'regression':
        critical_metric = 'accuracy'
    else:
        critical_metric = 'loss'

    model_weights_path = os.path.join(
        trainer.writer.experiment_folder,
        trainer.writer.experiment_title + '_BEST_val_{}.pth'.format(
            critical_metric))
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

    kwargs = sort_args(args.step, vars(args))
    trainer = Trainer(sets=['test'], **kwargs)
    trainer.testing()


if __name__ == '__main__':
    base_path = os.getcwd()
    setup_folders(base_path)
    args = get_arguments(base_path)

    init_distributed(args)
    model_weights_path, step, task = weight_loader(args)

    if step == '4':
        test(args, '4', model_weights_path)
    else:
        print(f'Starting phase {step}: {task}')
        run_phase(args, model_weights_path, step, task)
        print(f'Finished phase {step}: {task}')
