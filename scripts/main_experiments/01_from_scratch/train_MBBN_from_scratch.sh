#!/bin/bash

## 01 scripts explanation
# ABCD : name of dataset
# sex : name of task
# divfreqBERT : name of model (step : 2)
# seed1 : seed is set as 1. this decides splits

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

cd ../../../

## ABCD sex
python main.py --dataset_name ABCD --step 2 --batch_size_phase2 32 --lr_init_phase2 3e-5 \
--workers_phase2 16 --fine_tune_task binary_classification --target sex --intermediate_vec {set as ROI number} \
--fmri_type divided_timeseries --nEpochs_phase2 100 --transformer_hidden_layers 8 --num_heads 8 \
--exp_name from_scratch_seed1 --seed 1 --sequence_length_phase2 348 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log  --spatial_loss_factor 1.0\

## ABIDE ASD
python main.py --dataset_name ABIDE --step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 8 --fine_tune_task binary_classification --target ASD --intermediate_vec {set as ROI number} \
--fmri_type divided_timeseries --nEpochs_phase2 100 --transformer_hidden_layers 8 --num_heads 8 \
--exp_name from_scratch_seed11 --seed 1 --sequence_length_phase2 280 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 1.0 \

## UKB sex
python main.py --dataset_name UKB --step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 \
--workers_phase2 8 --fine_tune_task binary_classification --target ASD --intermediate_vec {set as ROI number} \
--fmri_type divided_timeseries --nEpochs_phase2 100 --transformer_hidden_layers 8 --num_heads 8 \
--exp_name from_scratch_seed1 --seed 1 --sequence_length_phase2 464 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \