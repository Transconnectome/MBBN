#!/bin/bash
#SBATCH --job-name=vanillaBERT_ABCD_HCPMMP1_ADHD_seed3      # Job name
#SBATCH --nodes=1                      # Total # of nodes
#SBATCH --nodelist=node3
#SBATCH --ntasks-per-node=1          # # of tasks per node
#SBATCH --cpus-per-task=16              # # of cores per task
#SBATCH --time=180:00:00                # Time limit (hh:mm:ss)
#SBATCH --output=/scratch/connectome/stellasybae/for_share/divfreqBERT/slurm_logs/%A_%x.out         # Standard error
#SBATCH --error=/scratch/connectome/stellasybae/for_share/divfreqBERT/slurm_logs/%A_%x.err          # Standard output
#SBATCH --mail-user=stellasybae@snu.ac.kr  # Email address
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --gpus-per-task=1


## 03 labserver version
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate divfreq
cd /scratch/connectome/stellasybae/for_share/divfreqBERT


python main.py --dataset_name ABCD --step 1 --batch_size_phase1 32 --lr_policy_phase1 SGDR \
--lr_init_phase1 3e-5 --weight_decay_phase1 1e-2 --lr_warmup_phase1 500 --lr_step_phase1 3000 --workers_phase1 8 \
--fine_tune_task binary_classification --target ADHD_label --intermediate_vec 180 --fmri_type timeseries --nEpochs_phase1 100 \
--transformer_hidden_layers 8 --num_heads 12 --exp_name 241019_vanilla_BERT_HCPMMP1_180_seed3 --distributed True --seed 3 \
--sequence_length_phase1 348 --seq_part head