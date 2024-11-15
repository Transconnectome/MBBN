#!/bin/bash
#SBATCH --job-name=vanillaBERT_ABIDE_Schaefer_ASD_seed5      # Job name
#SBATCH --nodes=1                      # Total # of nodes
#SBATCH --nodelist=node1
#SBATCH --ntasks-per-node=1          # # of tasks per node
#SBATCH --cpus-per-task=16              # # of cores per task
#SBATCH --time=05:00:00                # Time limit (hh:mm:ss)
#SBATCH --output=/scratch/connectome/stellasybae/for_share/divfreqBERT/slurm_logs/%A_%x.out         # Standard error
#SBATCH --error=/scratch/connectome/stellasybae/for_share/divfreqBERT/slurm_logs/%A_%x.err          # Standard output
#SBATCH --mail-user=stellasybae@snu.ac.kr  # Email address
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --gpus-per-task=1


## 03 labserver version
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate divfreq
cd /scratch/connectome/stellasybae/for_share/divfreqBERT


python main.py --dataset_name ABIDE --step 1 --batch_size_phase1 8 --lr_policy_phase1 step \
--lr_init_phase1 3e-5 --weight_decay_phase1 1e-2 --workers_phase1 8 \
--fine_tune_task binary_classification --target ASD --intermediate_vec 400 --fmri_type timeseries --nEpochs_phase1 100 \
--transformer_hidden_layers 8 --num_heads 8 --exp_name 241021_vanilla_BERT_Schaefer_seed5 --distributed True --seed 5 \
--sequence_length_phase1 280 --seq_part head