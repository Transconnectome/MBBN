#!/bin/bash
## 01 scripts explanation
# after pretraining, you can run main.py with --weightwatcher option

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}


## 03 experiment

python main.py --step 2 --fmri_type divided_timeseries --exp_name test --wandb_mode offline \
--transformer_hidden_layers 8 --num_heads 8 --exp_name {name of figure} \
--spatiotemporal --spat_diff_loss_type minus_log \
--pretrained_model_weights_path {your pretrained model path} \
--wandb_mode offline --weightwatcher \
--weightwatcher_save_dir {the path you want to store weightwatcher results} \
