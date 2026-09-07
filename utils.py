from data_preprocess_and_load.datasets import * #####
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datetime import datetime
from pytz import timezone
import argparse
import os
import dill
import random
import builtins
import time
import random
import warnings

def _get_sync_file():    
        """Logic for naming sync file using slurm env variables"""
        if 'SCRATCH' in os.environ:
            sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH'] # Perlmutter
        else:
            sync_file_dir = '%s/pytorch-sync-files' % '/lus/grand/projects/STlearn/'
            #raise Exception('there is no env variable SCRATCH. Please check sync_file dir')
        os.makedirs(sync_file_dir, exist_ok=True)

        #temporally add two lines below for torchrun
        if ('SLURM_JOB_ID' in os.environ) and ('SLURM_STEP_ID' in os.environ) :
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (
            sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        else:
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (
            sync_file_dir, '10004', '10003')
        return sync_file
    
    
def init_distributed(args):   
    
    # torchrun: when WORLD_SIZE is set in the sbatch script (gpus_per_node * num_nodes)
    if "WORLD_SIZE" in os.environ: # for torchrun
        args.world_size = int(os.environ["WORLD_SIZE"])
        #print('args.world_size:',args.world_size)
    elif 'SLURM_NTASKS' in os.environ: # for slurm scheduler
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        pass # torch.distributed.launch
        
    args.distributed = args.world_size > 1 # default: world_size = -1 
    
    if args.distributed:
        start_time = time.time()
        #args.local_rank = int(os.environ['LOCAL_RANK']) #stella added this line
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'RANK' in os.environ: # for torchrun
            args.rank = int(os.environ['RANK'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        
        #print('args.rank:',args.rank)
        #print('args.gpu:',args.gpu)    
        if args.init_method == 'file':
            sync_file = _get_sync_file()
            # print('initializing DDP with sync file')
        elif args.init_method == 'env':
            sync_file = "env://"
            #os.environ['MASTER_PORT'] = '47769'
            #os.environ['MASTER_ADDR'] = '127.0.0.1'# os.environ['SLURM_JOB_NODELIST']
            #print(os.environ['MASTER_PORT'])
            #print(os.environ['MASTER_ADDR'])
            #print('initializing DDP with env variables')
        dist.init_process_group(backend=args.dist_backend, init_method=sync_file,
                            world_size=args.world_size, rank=args.rank)
        #dist_init_time = time.time() - start_time
        #print(f'seconds taken for DDP initialization: {dist_init_time}')
    else:
        args.rank = 0
        args.gpu = 0

        
_TASK_BY_STEP = {'1': 'vanilla_BERT', '2': 'MBBN', '3': 'MBBN_pretraining', '4': 'test'}


def weight_loader(args):
    """Resolve (initial weights, step, task name) for the requested phase.

    Only paths the parser actually defines are consulted, and a path that was
    given but does not exist is an error rather than a silent fall-through to
    random initialisation.
    """
    step = str(args.step)
    if step not in _TASK_BY_STEP:
        raise ValueError(f'unknown step {step!r}')
    task = _TASK_BY_STEP[step]

    candidate = getattr(args, 'model_weights_path_phase' + step, None)
    if candidate is None:
        candidate = getattr(args, 'pretrained_model_weights_path', None)
    if candidate in (None, '', 'None'):
        return None, step, task
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f'--model_weights_path_phase{step} / --pretrained_model_weights_path '
            f'points at {candidate!r}, which does not exist. Fix the path or omit '
            f'the flag to train from scratch.')
    return candidate, step, task
    
def datestamp():
    time = datetime.now(timezone('Asia/Seoul')).strftime("%m_%d__%H_%M_%S")
    return time

def reproducibility(**kwargs):
    """Seed every RNG that affects training.

    `deterministic=True` additionally pins cuDNN and asks torch for
    deterministic kernels, which makes a run bit-reproducible at some cost in
    throughput. The published default (`False`) is kept so existing runs are
    unchanged.
    """
    seed = kwargs.get('seed')
    cuda = kwargs.get('cuda')
    deterministic = bool(kwargs.get('deterministic', False))
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def split_seed_of(kwargs):
    """Seed governing the train/val/test partition.

    Defaults to `seed` so existing split files keep their names; set
    `--split_seed` to hold the partition fixed while varying weight init.
    """
    s = kwargs.get('split_seed')
    return kwargs.get('seed') if s is None else s

def sort_args(phase, args, explicit=None):
    """Keep phase-agnostic arguments plus those tagged with the active phase.

    `explicit` is the set of option names the user actually typed. Any of those
    tagged for a *different* phase is dropped here, which silently reverts the
    setting to its default; warn so that never happens unnoticed again.
    """
    phase = str(phase)
    phase_specific_args = {}
    dropped = []
    for name, value in args.items():
        if not 'phase' in name:
            phase_specific_args[name] = value
        elif 'phase' + phase in name:
            phase_specific_args[name.replace('_phase' + phase, '')] = value
        elif explicit and name in explicit:
            dropped.append(name)
    if dropped:
        warnings.warn(
            'these options were given on the command line but belong to another '
            'phase, so they have NO effect on step {}: {}. Rename them to '
            '*_phase{} if that is what you meant.'.format(phase, sorted(dropped), phase),
            RuntimeWarning, stacklevel=2)
    return phase_specific_args

def args_logger(args):
    args_to_pkl(args)
    args_to_text(args)


def args_to_pkl(args):
    with open(os.path.join(args.experiment_folder,'arguments_as_is.pkl'),'wb') as f:
        #f.write(vars(args))
        dill.dump(vars(args),f)

def args_to_text(args):
    with open(os.path.join(args.experiment_folder,'argument_documentation.txt'),'w+') as f:
        for name,arg in vars(args).items():
            f.write('{}: {}\n'.format(name,arg))
