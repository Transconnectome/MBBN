#!/bin/bash

## 01 scripts explanation
# large : large model (hidden layers 8, number of heads 12)
# mask01 : Use mask loss whose masking rate is 0.1
# recon : Use reconstruction loss

## 02 perlmutter version
module load pytorch/1.13.1
cd /global/cfs/projectdirs/m4244/stella/divfreqBERT

## 03 labserver version
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

python main.py --dataset_name ABCD --step 5 --batch_size_phase5 32 --lr_policy_phase5 SGDR --lr_init_phase5 1e-3 --lr_warmup_phase5 500 --target reconstruction --lr_step_phase5 1000 --workers_phase5 0 --intermediate_vec 180 --nEpochs_phase5 200 --transformer_hidden_layers 8 --num_heads_mult 12 --fmri_type divided_timeseries --seed 1 --exp_name large_masking010_recon --transformer_dropout_rate 0.3 --distributed True --use_mask_loss --use_recon_loss --divide_by_lorentzian --low_ultralow_rate 0.1 --masking_rate 0.1