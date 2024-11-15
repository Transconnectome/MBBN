#!/bin/bash

## 01 scripts explanation
# ABCD : name of dataset
# sex : name of task
# divfreqBERT : name of model (step : 2)
# seed1 : seed is set as 1. this influences splits

## 02 perlmutter version
module load pytorch/1.13.1
cd /global/cfs/projectdirs/m4244/stella/divfreqBERT

## 03 labserver version
# conda activate {your environment}
# cd {your directory which contains 'main.py'}


python main.py --dataset_name ABCD --step 2 --batch_size_phase2 16 --lr_policy_phase2 SGDR --lr_init_phase2 3e-5 --weight_decay_phase2 1e-2 --lr_warmup_phase2 500 --lr_step_phase2 3000 --workers_phase2 16 --fine_tune_task binary_classification --feature_map_size same --feature_map_gen no --target sex  --intermediate_vec 180 --fmri_type timeseries_and_frequency --nEpochs_phase2 200 --concat_method concat --fmri_multimodality_type two_channels --filtering_type FIR --transformer_hidden_layers 8 --num_heads_mult 12 --exp_name 230524_timeseries_and_frequency_lr3e5_seed1 --transformer_dropout_rate 0.3 --distributed True --seed 1 --divide_by_lorentzian --low_ultralow_rate 0.1