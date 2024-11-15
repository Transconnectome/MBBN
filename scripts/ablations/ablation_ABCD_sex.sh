#!/bin/bash

## 01 scripts explanation
# ABCD : name of dataset
# sex : name of task
# divfreqBERT : name of model (step : 2)
# seed1 : seed is set as 1. this decides splits

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

# ablation 1 : no frequency division
python main.py --dataset_name ABCD --step 1 --batch_size_phase1 32 --lr_policy_phase1 SGDR \
--lr_init_phase1 3e-5 --weight_decay_phase1 1e-2 --lr_warmup_phase1 500 --lr_step_phase1 3000 --workers_phase1 8 \
--fine_tune_task binary_classification --target sex --intermediate_vec {set as ROI number} --fmri_type timeseries --nEpochs_phase1 100 \
--transformer_hidden_layers 8 --num_heads 8 --exp_name NO_frequency_division_seed1 --distributed True --seed 1 \
--sequence_length_phase1 348 --seq_part head

# ablation 2 : only high / low / ultralow
python main.py --dataset_name ABCD --step 1 --batch_size_phase1 32 --lr_policy_phase1 SGDR \
--lr_init_phase1 3e-5 --weight_decay_phase1 1e-2 --lr_warmup_phase1 500 --lr_step_phase1 3000 --workers_phase1 8 \
--fine_tune_task binary_classification --target sex --intermediate_vec {set as ROI number} --fmri_type time_domain_ultralow --nEpochs_phase1 100 \
--transformer_hidden_layers 8 --num_heads 8 --exp_name NO_frequency_division_seed1 --distributed True --seed 1 \
--sequence_length_phase1 348 --seq_part head

# ablation 3 : two frequencies
python main.py --dataset_name ABCD --step 2 --batch_size_phase2 32 --lr_policy_phase1 SGDR \
--lr_init_phase2 3e-5 --weight_decay_phase2 1e-2 --lr_warmup_phase2 500 --lr_step_phase2 3000 --workers_phase2 8 \
--fine_tune_task binary_classification --target sex --intermediate_vec {set as ROI number} --fmri_type divided_timeseries --nEpochs_phase2 100 \
--transformer_hidden_layers 8 --num_heads 8 --exp_name NO_frequency_division_seed1 --distributed True --seed 1 \
--sequence_length_phase2 348 --seq_part head --use_raw_knee --fmri_multimodality_type two_channels

# ablation 4 : convolution
python main.py --dataset_name ABCD --step 2 --batch_size_phase2 32 --lr_policy_phase1 SGDR \
--lr_init_phase2 3e-5 --weight_decay_phase2 1e-2 --lr_warmup_phase2 500 --lr_step_phase2 3000 --workers_phase2 8 \
--fine_tune_task binary_classification --target sex --intermediate_vec {set as ROI number} --fmri_type divided_timeseries --nEpochs_phase2 100 \
--transformer_hidden_layers 8 --num_heads 8 --exp_name NO_frequency_division_seed1 --distributed True --seed 1 \
--sequence_length_phase2 348 --seq_part head --use_raw_knee --fmri_multimodality_type three_channels --use_high_freq \
--ablation convolution