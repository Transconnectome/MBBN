#!/bin/bash
#SBATCH --job-name=communicability
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%A_%x.out
#SBATCH --error=slurm_logs/%A_%x.err

# Compute communicability-ordered ROI lists for pretraining masking.
# Run this once before pretraining. Results are saved to data/communicability/.
#
# ROI_num: 400 for Schaefer atlas, 360 for HCP-MMP1

ROI_NUM=400       # 400 (Schaefer) | 360 (HCP-MMP1)
DATASET=UKB       # UKB (pretraining dataset)

source /usr/anaconda3/etc/profile.d/conda.sh
conda activate mbbn
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

python communicability.py --dataset_name ${DATASET} --ROI_num ${ROI_NUM}
