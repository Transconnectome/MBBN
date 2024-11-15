from utils import *  #including 'init_distributed', 'weight_loader'
from trainer import Trainer
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from pathlib import Path


def get_arguments(base_path):
    """
    handle arguments from commandline.
    some other hyper parameters can only be changed manually (such as model architecture,dropout,etc)
    notice some arguments are global and take effect for the entire three phase training process, while others are determined per phase
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default="baseline") 
    parser.add_argument('--dataset_name', type=str, choices=['HCP1200', 'ABCD', 'ABIDE', 'UKB'], default="ABCD")
    parser.add_argument('--fmri_type', type=str, choices=['timeseries', 'frequency', 'divided_timeseries', 'time_domain_low', 'time_domain_ultralow', 'time_domain_high', 'frequency_domain_low', 'frequency_domain_ultralow', 'frequency_domain_high'], default="divided_timeseries")
    parser.add_argument('--intermediate_vec', type=int, default=400)
    parser.add_argument('--abcd_path', default='/scratch/connectome/stellasybae/ABCD_ROI/7.ROI') ## labserver
    parser.add_argument('--ukb_path', default='/scratch/connectome/stellasybae/UKB_ROI') ## labserver
    parser.add_argument('--abide_path', default='/scratch/connectome/stellasybae/ABIDE_ROI') ## labserver
    parser.add_argument('--base_path', default=base_path) # where your main.py, train.py, model.py are in.
    parser.add_argument('--step', default='1', choices=['1','2','3','4'], help='which step you want to run')
    
    
    parser.add_argument('--target', type=str, default='sex')
    parser.add_argument('--fine_tune_task',
                        choices=['regression','binary_classification'],
                        help='fine tune model objective. choose binary_classification in case of a binary classification task')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--prepare_visualization', action='store_true')
    parser.add_argument('--weightwatcher', action='store_true')
    parser.add_argument('--weightwatcher_save_dir', default=None)

    
    
    parser.add_argument('--norm_axis', default=1, type=int, choices=[0,1,None])
    
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--log_dir', type=str, default=os.path.join(base_path, 'runs'))

    parser.add_argument('--transformer_hidden_layers', type=int,default=8)
    
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--init_method', default='file', type=str, choices=['file','env'], help='DDP init method')
    parser.add_argument('--distributed', default=True)

    # AMP configs:
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--gradient_clipping', action='store_true')
    parser.add_argument('--clip_max_norm', type=float, default=1.0)
    
    # Gradient accumulation
    parser.add_argument("--accumulation_steps", default=1, type=int,required=False,help='mini batch size == accumulation_steps * args.train_batch_size')
    
    # Nsight profiling
    parser.add_argument("--profiling", action='store_true')
    
    #wandb related
    parser.add_argument('--wandb_key', default='108101f4b9c3e31a235aa58307d1c6b548cfb54a', type=str,  help='default: key for Stella')
    parser.add_argument('--wandb_mode', default='online', type=str,  help='online|offline')
    parser.add_argument('--wandb_entity', default='stellasybae', type=str)
    parser.add_argument('--wandb_project', default='divfreqbert', type=str)

    
    # dividing
    parser.add_argument('--filtering_type', default='Boxcar', choices=['FIR', 'Boxcar'])
    parser.add_argument('--use_high_freq', action='store_true')
    parser.add_argument('--divide_by_lorentzian', action='store_true')
    parser.add_argument('--use_raw_knee', action='store_true')
    parser.add_argument('--seq_part', type=str, default='head')
    parser.add_argument('--fmri_dividing_type', default='three_channels', choices=['two_channels', 'three_channels'])
    
    # Dropouts
    parser.add_argument('--transformer_dropout_rate', type=float, default=0.3) 

    # Architecture
    parser.add_argument('--num_heads', type=int, default=12,
                        help='number of heads for BERT network (default: 12)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
                        
    
    ## for finetune
    parser.add_argument('--pretrained_model_weights_path', default=None)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_test', action='store_true', help='test phase of finetuning task')
    
    
    ## spatiotemporal
    parser.add_argument('--spatiotemporal', action = 'store_true')
    parser.add_argument('--spat_diff_loss_type', type=str, default='minus_log', choices=['minus_log', 'reciprocal_log', 'exp_minus', 'log_loss', 'exp_whole'])
    parser.add_argument('--spatial_loss_factor', type=float, default=0.1)
    
    ## ablation
    parser.add_argument('--ablation', type=str, choices=['convolution', 'no_high_freq'])
    
    ## phase 1 vanilla BERT
    parser.add_argument('--task_phase1', type=str, default='vanilla_BERT')
    parser.add_argument('--batch_size_phase1', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples')
    parser.add_argument('--validation_frequency_phase1', type=int, default=10000000)
    parser.add_argument('--nEpochs_phase1', type=int, default=100)
    parser.add_argument('--optim_phase1', default='AdamW')
    parser.add_argument('--weight_decay_phase1', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase1', default='SGDR', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase1', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase1', type=float, default=0.97)
    parser.add_argument('--lr_step_phase1', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase1', type=int, default=500)
    parser.add_argument('--sequence_length_phase1', type=int ,default=348) # ABCD 348 ABIDE 280 UKB 464
    parser.add_argument('--workers_phase1', type=int,default=4)
    parser.add_argument('--num_heads_2DBert', type=int, default=12)
    
    ## phase 2 MBBN
    parser.add_argument('--task_phase2', type=str, default='MBBN')
    parser.add_argument('--batch_size_phase2', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples')
    parser.add_argument('--nEpochs_phase2', type=int, default=100)
    parser.add_argument('--optim_phase2', default='AdamW')
    parser.add_argument('--weight_decay_phase2', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase2', default='SGDR', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase2', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase2', type=float, default=0.97)
    parser.add_argument('--lr_step_phase2', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase2', type=int, default=500)
    parser.add_argument('--sequence_length_phase2', type=int ,default=348) # ABCD 348 ABIDE 280 UKB 464
    parser.add_argument('--workers_phase2', type=int, default=4)
    
    ##phase 3 pretraining
    parser.add_argument('--task_phase3', type=str, default='MBBN_pretraining')
    parser.add_argument('--batch_size_phase3', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples')
    parser.add_argument('--validation_frequency_phase3', type=int, default=10000000)
    parser.add_argument('--nEpochs_phase3', type=int, default=1000)
    parser.add_argument('--optim_phase3', default='AdamW')
    parser.add_argument('--weight_decay_phase3', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase3', default='SGDR', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase3', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase3', type=float, default=0.97)
    parser.add_argument('--lr_step_phase3', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase3', type=int, default=500)
    parser.add_argument('--sequence_length_phase3', type=int ,default=464)
    parser.add_argument('--workers_phase3', type=int,default=4)
    parser.add_argument('--use_recon_loss', action='store_true')
    parser.add_argument('--use_mask_loss', action='store_true') 
    parser.add_argument('--use_cont_loss', action='store_true')
    parser.add_argument('--masking_rate', type=float, default=0.1)
    parser.add_argument('--masking_method', type=str, default='spatiotemporal', choices=['temporal', 'spatial', 'spatiotemporal'])
    parser.add_argument('--temporal_masking_type', type=str, default='time_window', choices=['single_point','time_window'])
    parser.add_argument('--temporal_masking_window_size', type=int, default=20)
    parser.add_argument('--window_interval_rate', type=int, default=2)
    parser.add_argument('--spatial_masking_type', type=str, default='hub_ROIs', choices=['hub_ROIs', 'random_ROIs'])
    parser.add_argument('--communicability_option', type=str, default='remove_high_comm_node', choices=['remove_high_comm_node', 'remove_low_comm_node'])
    parser.add_argument('--num_hub_ROIs', type=int, default=5)
    parser.add_argument('--num_random_ROIs', type=int, default=5)
    parser.add_argument('--spatiotemporal_masking_type', type=str, default='whole', choices=['whole', 'separate'])
    
    
    ## phase 4 (test)
    parser.add_argument('--task_phase4', type=str, default='test')
    parser.add_argument('--model_weights_path_phase4', default=None)
    parser.add_argument('--batch_size_phase4', type=int, default=4)
    parser.add_argument('--nEpochs_phase4', type=int, default=1)
    parser.add_argument('--optim_phase4', default='AdamW')
    parser.add_argument('--weight_decay_phase4', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase4', default='SGDR', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase4', type=float, default=1e-4)
    parser.add_argument('--lr_gamma_phase4', type=float, default=0.9)
    parser.add_argument('--lr_step_phase4', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase4', type=int, default=100)
    parser.add_argument('--sequence_length_phase4', type=int,default=348) # ABCD 348 ABIDE 280 UKB 464
    parser.add_argument('--workers_phase4', type=int, default=4)
                        
    args = parser.parse_args()
        
    return args

def setup_folders(base_path): 
    os.makedirs(os.path.join(base_path,'experiments'),exist_ok=True) 
    os.makedirs(os.path.join(base_path,'runs'),exist_ok=True)
    os.makedirs(os.path.join(base_path, 'splits'), exist_ok=True)
    return None

def run_phase(args,loaded_model_weights_path,phase_num,phase_name):
    """
    main process that runs each training phase
    :return path to model weights (pytorch file .pth) aquried by the current training phase
    """
    experiment_folder = '{}_{}_{}_{}'.format(args.dataset_name,phase_name,args.target,args.exp_name)
    experiment_folder = Path(os.path.join(args.base_path,'experiments',experiment_folder))
    os.makedirs(experiment_folder, exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num,loaded_model_weights_path)
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    
    print(f'saving the results at {args.experiment_folder}')
    
    # save hyperparameters
    args_logger(args)
    
    # make args to dict. + detach phase numbers from args
    kwargs = sort_args(phase_num, vars(args))
    if args.prepare_visualization:
        S = ['train','val']
    else:
        S = ['train','val','test']

    trainer = Trainer(sets=S,**kwargs)
    trainer.training()

    #S = ['train','val']

    if phase_num == '3' and not fine_tune_task == 'regression':
        critical_metric = 'accuracy'
    else:
        critical_metric = 'loss'
    model_weights_path = os.path.join(trainer.writer.experiment_folder,trainer.writer.experiment_title + '_BEST_val_{}.pth'.format(critical_metric)) 

    return model_weights_path


def test(args,phase_num,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), args.exp_name) #, datestamp())
    experiment_folder = Path(os.path.join(args.base_path,'tests', experiment_folder))
    os.makedirs(experiment_folder,exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num, model_weights_path)
    
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    args_logger(args)
    args = sort_args(args.step, vars(args))
    S = ['test']
    trainer = Trainer(sets=S,**args)
    trainer.testing()

if __name__ == '__main__':
    base_path = os.getcwd() 
    setup_folders(base_path) 
    args = get_arguments(base_path)

    # DDP initialization
    init_distributed(args)

    # load weights that you specified at the Argument
    model_weights_path, step, task = weight_loader(args)

    if step == '4' :
        print(f'starting testing')
        phase_num = '4'
        test(args, phase_num, model_weights_path)
    else:
        print(f'starting phase{step}: {task}')
        run_phase(args,model_weights_path,step,task)
        print(f'finishing phase{step}: {task}')
