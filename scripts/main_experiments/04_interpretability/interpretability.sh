#!/bin/bash
## 01 scripts explanation
# after finetuning or training-from-scratch using --prepare_visualization option, you can run visualization.py

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'visualization.py'}


## 03 experiment
# 3-1. ABCD ADHD
python visualization.py --dataset_name ABCD --step 3 --batch_size 1 \
--fine_tune_task binary_classification --target ADHD_label --intermediate_vec {set as ROI number} --fmri_type divided_timeseries \
--transformer_hidden_layers 8 --num_heads 8 --filtering_type Boxcar --exp_name test --wandb_mode offline \
--seed 1 --sequence_length 348 --divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels \
--use_high_freq --spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 1.0 --finetune \
--model_path {your finetuned model path} \
--save_dir {the path you want to store interpretability results} \

# 3-2. ABIDE ASD
python visualization.py --dataset_name ABIDE --step 3 --batch_size 1 \
--fine_tune_task binary_classification --target ASD --intermediate_vec {set as ROI number} --fmri_type divided_timeseries \
--transformer_hidden_layers 8 --num_heads 8 --filtering_type Boxcar --exp_name test --wandb_mode offline --lr_policy step \
--seed 1 --sequence_length 280 --divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels \
--use_high_freq --spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 1.0  --finetune \
--model_path {your finetuned model path} \
--save_dir {the path you want to store interpretability results} \