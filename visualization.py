import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

import os
from tqdm import tqdm
import json
import numpy as np

from pathlib import Path

import sys

from model import *
import argparse
from trainer import *
from data_preprocess_and_load.dataloaders import *


## ABIDE!

def get_arguments(base_path = os.getcwd()):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default="baseline") 
    parser.add_argument('--dataset_name', type=str, choices=['HCP1200', 'ABCD', 'ABIDE', 'UKB'], default="ABCD")
    parser.add_argument('--fmri_type', type=str, choices=['timeseries', 'frequency', 'divided_timeseries', 'time_domain_low', 'time_domain_ultralow', 'time_domain_high' , 'frequency_domain_low', 'frequency_domain_ultralow', 'frequency_domain_high'], default="divided_timeseries")
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
    #parser.add_argument('--distributed', default=True)

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
    parser.add_argument('--seq_part', type=str, default='tail')
    parser.add_argument('--fmri_dividing_type', default='three_channels', choices=['two_channels', 'three_channels'])

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
    
    ##phase 3 (test)
    parser.add_argument('--transformer_dropout_rate', type=float, default=0.3)
    parser.add_argument('--task', type=str, default='test')
    parser.add_argument('--model_weights_path_phase2', default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=20)
    parser.add_argument('--augment_prob', default=0)
    parser.add_argument('--optim', default='AdamW')
    parser.add_argument('--weight_decay3', type=float, default=1e-2)
    parser.add_argument('--lr_policy', default='step', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--lr_gamma', type=float, default=0.9)
    parser.add_argument('--lr_step', type=int, default=1500)
    parser.add_argument('--lr_warmup', type=int, default=100)
    parser.add_argument('--sequence_length', type=int,default=348)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_dir', type=str)
    
    args = parser.parse_args()
        
    return args


args = get_arguments()
model_path = args.model_path
dataset_name = args.dataset_name
model = Transformer_Finetune_Three_Channels(**vars(args))
state_dict = torch.load(model_path)['model_state_dict']
model.load_state_dict(state_dict)
model.eval()
model.cuda(0) if torch.cuda.is_available() else model

if dataset_name == 'ABIDE':
    lower_bound = 0.45
    upper_bound = 0.55
elif dataset_name == 'ABCD':
    lower_bound = 0.75
    upper_bound = 0.25
    
def get_activation(dict, name):
    def hook(model, input, output):
        dict[name] = output[0].detach().tolist()
    return hook


# integrated_gradients = IntegratedGradients(model)
# noise_tunnel = NoiseTunnel(integrated_gradients)

data_handler = DataHandler(**vars(args))
_, _, test_loader = data_handler.create_dataloaders()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
dataset_name = args.dataset_name
kwargs = {
    "nt_samples": 5,
    "nt_samples_batch_size": 5,
    "nt_type": "smoothgrad_sq", # 1
    #"stdevs": 0.05,
    "internal_batch_size": 5,
}


for idx, data in enumerate(tqdm(test_loader),0):
    subj_name = data['subject_name'][0]
    
    # input itself
    input_low = data['fmri_lowfreq_sequence'].float().requires_grad_(True).cuda(0)
    input_ultralow = data['fmri_ultralowfreq_sequence'].requires_grad_(True).float().cuda(0)
    input_high = data['fmri_highfreq_sequence'].float().requires_grad_(True).cuda(0)
    
    label = data[args.target].float().cuda(0)
    pred = model(input_high, input_low, input_ultralow)[args.fine_tune_task]
    pred_prob = torch.sigmoid(pred)
    pred_int = (pred_prob>0.5).int().item()
    target_int = label.int().item()
    
    #only choose corrected samples
    
    if pred_int == target_int:
        if target_int == 0:
            if pred_prob <= lower_bound:
                # target 0
                file_dir = os.path.join(save_dir, f'{dataset_name}_target0')
                os.makedirs(file_dir,exist_ok=True)
                
                # 01 filename
                file_path_input_low = os.path.join(file_dir, f"{subj_name}_input_low.pt")
                file_path_input_ultralow = os.path.join(file_dir, f"{subj_name}_input_ultralow.pt")
                file_path_input_high = os.path.join(file_dir, f"{subj_name}_input_high.pt")
                activation_path = os.path.join(file_dir, f"{subj_name}_att_mat_activation.json")
                gradient_path = os.path.join(file_dir, f"{subj_name}_att_mat_gradient.json")

                # 02-2 noise tunnel - att mat
                # register forward hook for the layer
                activation_layer_high = 'high_spatial_attention' 
                activation_layer_low = 'low_spatial_attention' 
                activation_layer_ultralow = 'ultralow_spatial_attention' 

                activations = {}
                model.high_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_high))
                model.low_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_low))
                model.ultralow_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_ultralow))

                # register backward hook for the layer
                gradients = {}
                model.high_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_high))
                model.low_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_low))
                model.ultralow_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_ultralow))

                # forward
                h = model.high_spatial_attention(input_high.permute(0, 2, 1)).float().cuda(0)
                l = model.low_spatial_attention(input_low.permute(0, 2, 1)).float().cuda(0)
                u = model.ultralow_spatial_attention(input_ultralow.permute(0, 2, 1)).float().cuda(0)

                # backward
                loss_fn = nn.L1Loss()
                spat_diff_loss = -torch.log((loss_fn(h, l)+loss_fn(h, u)+loss_fn(l, u)))
                spat_diff_loss.backward()
                
                
                with open(activation_path, 'w') as f : 
                    json.dump(activations, f, indent=4)
                with open(gradient_path, 'w') as f : 
                    json.dump(gradients, f, indent=4)
                        
                print(f'saving {subj_name}')
        
        elif target_int == 1:
            if pred_prob >= upper_bound:
                # target 1
                file_dir = os.path.join(save_dir, f'{dataset_name}_target1')
                os.makedirs(file_dir,exist_ok=True)
                # 01 filename
                file_path_input_low = os.path.join(file_dir, f"{subj_name}_input_low.pt")
                file_path_input_ultralow = os.path.join(file_dir, f"{subj_name}_input_ultralow.pt")
                file_path_input_high = os.path.join(file_dir, f"{subj_name}_input_high.pt")
                activation_path = os.path.join(file_dir, f"{subj_name}_att_mat_activation.json")
                gradient_path = os.path.join(file_dir, f"{subj_name}_att_mat_gradient.json")

                # 02-2 noise tunnel - att mat
                # register forward hook for the layer
                activation_layer_high = 'high_spatial_attention'  
                activation_layer_low = 'low_spatial_attention'  
                activation_layer_ultralow = 'ultralow_spatial_attention'  

                activations = {}
                model.high_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_high))
                model.low_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_low))
                model.ultralow_spatial_attention.register_forward_hook(get_activation(activations, activation_layer_ultralow))

                # register backward hook for the layer
                gradients = {}
                model.high_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_high))
                model.low_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_low))
                model.ultralow_spatial_attention.register_backward_hook(get_activation(gradients, activation_layer_ultralow))

                # forward
                h = model.high_spatial_attention(input_high.permute(0, 2, 1)).float().cuda(0)
                l = model.low_spatial_attention(input_low.permute(0, 2, 1)).float().cuda(0)
                u = model.ultralow_spatial_attention(input_ultralow.permute(0, 2, 1)).float().cuda(0)

                # backward
                loss_fn = nn.L1Loss()
                spat_diff_loss = -torch.log((loss_fn(h, l)+loss_fn(h, u)+loss_fn(l, u)))
                spat_diff_loss.backward()
                
                # activation = activations[activation_layer]
                # gradient = gradients[activation_layer]
                
                with open(activation_path, 'w') as f : 
                    json.dump(activations, f, indent=4)
                with open(gradient_path, 'w') as f : 
                    json.dump(gradients, f, indent=4)
                
                print(f'saving {subj_name}')