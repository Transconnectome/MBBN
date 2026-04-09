#!/bin/bash
#SBATCH --job-name=MBBN_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=180:00:00
#SBATCH --output=slurm_logs/%A_%x.out
#SBATCH --error=slurm_logs/%A_%x.err

# ── User settings ────────────────────────────────────────────────────────────
SEED=1
EXP_NAME="MBBN_finetune"

# Dataset / task
DATASET=ABIDE                   # ABCD | UKB | ABIDE
TARGET=ASD                      # sex | ADHD_label | ASD | fluid_intelligence
FINE_TUNE_TASK=binary_classification   # binary_classification | regression

# Atlas / architecture
#   HCP-MMP1 (360 ROIs): --intermediate_vec 360 --num_heads 12
#   Schaefer (400 ROIs): --intermediate_vec 400 --num_heads 8
INTERMEDIATE_VEC=400
NUM_HEADS=8

# Sequence length: ABCD=348, UKB=464, ABIDE=280
SEQ_LEN=280

# Spatial loss factor λ (Table 7): ABCD→10, UKB→1, ABIDE→100
SPATIAL_LOSS_FACTOR=100.0

# Pretrained weights (from step 3 pretraining)
PRETRAINED_WEIGHTS="/path/to/pretrained_model.pth"

# ── Environment ──────────────────────────────────────────────────────────────
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate mbbn
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

# ── Run ──────────────────────────────────────────────────────────────────────
python main.py \
    --dataset_name ${DATASET} \
    --step 2 \
    --finetune \
    --fine_tune_task ${FINE_TUNE_TASK} \
    --target ${TARGET} \
    --intermediate_vec ${INTERMEDIATE_VEC} \
    --num_heads ${NUM_HEADS} \
    --sequence_length_phase2 ${SEQ_LEN} \
    --spatial_loss_factor ${SPATIAL_LOSS_FACTOR} \
    --pretrained_model_weights_path ${PRETRAINED_WEIGHTS} \
    --nEpochs_phase2 50 \
    --batch_size_phase2 16 \
    --lr_init_phase2 3e-5 \
    --seed ${SEED} \
    --exp_name ${EXP_NAME}
