#!/bin/bash
#SBATCH --job-name=MBBN_from_scratch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%A_%x.out
#SBATCH --error=slurm_logs/%A_%x.err

# ── User settings ────────────────────────────────────────────────────────────
SEED=1
EXP_NAME="MBBN_from_scratch"

# Dataset / task
DATASET=ABCD                    # ABCD | UKB | ABIDE
TARGET=sex                      # sex | ADHD_label | ASD | fluid_intelligence
FINE_TUNE_TASK=binary_classification   # binary_classification | regression

# Atlas / architecture
#   HCP-MMP1 (360 ROIs): --intermediate_vec 360 --num_heads 12
#   Schaefer (400 ROIs): --intermediate_vec 400 --num_heads 8
INTERMEDIATE_VEC=360
NUM_HEADS=12

# Sequence length: ABCD=348, UKB=464, ABIDE=280
SEQ_LEN=348

# Spatial loss factor λ (Table 7): ABCD→10, UKB→1, ABIDE→100
SPATIAL_LOSS_FACTOR=10.0

# Filtering: FIR for ABCD+HCP-MMP1 (Table 7); Boxcar otherwise
FILTERING_TYPE=FIR

# ── Environment ──────────────────────────────────────────────────────────────
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate mbbn
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

# ── Run ──────────────────────────────────────────────────────────────────────
python main.py \
    --dataset_name ${DATASET} \
    --step 2 \
    --fine_tune_task ${FINE_TUNE_TASK} \
    --target ${TARGET} \
    --intermediate_vec ${INTERMEDIATE_VEC} \
    --num_heads ${NUM_HEADS} \
    --sequence_length_phase2 ${SEQ_LEN} \
    --spatial_loss_factor ${SPATIAL_LOSS_FACTOR} \
    --filtering_type ${FILTERING_TYPE} \
    --nEpochs_phase2 100 \
    --batch_size_phase2 32 \
    --lr_init_phase2 3e-5 \
    --seed ${SEED} \
    --exp_name ${EXP_NAME}
