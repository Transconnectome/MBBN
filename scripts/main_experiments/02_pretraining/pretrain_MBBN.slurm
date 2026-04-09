#!/bin/bash
#SBATCH --job-name=MBBN_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=20000:00:00
#SBATCH --output=slurm_logs/%A_%x.out
#SBATCH --error=slurm_logs/%A_%x.err

# ── User settings ────────────────────────────────────────────────────────────
SEED=1
EXP_NAME="MBBN_pretrain"

# Pretraining is always on UKB with target=reconstruction.
# Atlas / architecture:
#   Schaefer (400 ROIs): --intermediate_vec 400 --num_heads 8   → 1000 epochs
#   HCP-MMP1 (360 ROIs): --intermediate_vec 360 --num_heads 12  → 400 epochs
INTERMEDIATE_VEC=400
NUM_HEADS=8
N_EPOCHS=1000

# Spatial loss factor λ: UKB → 1.0 (Table 7)
SPATIAL_LOSS_FACTOR=1.0

# ── Environment ──────────────────────────────────────────────────────────────
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate mbbn
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

# ── Run ──────────────────────────────────────────────────────────────────────
python main.py \
    --dataset_name UKB \
    --step 3 \
    --target reconstruction \
    --intermediate_vec ${INTERMEDIATE_VEC} \
    --num_heads ${NUM_HEADS} \
    --sequence_length_phase4 464 \
    --nEpochs_phase3 ${N_EPOCHS} \
    --batch_size_phase3 4 \
    --lr_init_phase3 3e-5 \
    --num_hub_ROIs 380 \
    --spatial_loss_factor ${SPATIAL_LOSS_FACTOR} \
    --seed ${SEED} \
    --exp_name ${EXP_NAME}
