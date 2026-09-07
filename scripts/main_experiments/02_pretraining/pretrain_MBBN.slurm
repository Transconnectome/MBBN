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

# ── Hardening (docs/audit/AUDIT.md) ──────────────────────────────────────────
# NOTE: this script used to pass `--sequence_length_phase4 464` while running
# step 3. sort_args() keeps only *_phase3 options for step 3, so that flag was
# discarded and pretraining ran at the phase-3 default (348) -- after which
# loading the checkpoint into a 464-wide fine-tuning model raises RuntimeError.
#
#   --random_mask                 resample the mask per subject per epoch
#                                 (published: one fixed mask for everyone, and
#                                  380/400 ROIs + every other window = ~97% of
#                                  the input hidden)
#   --mask_loss_on_masked_only    score reconstruction only where input was hidden
#   --cache_bands                 memoise the per-subject knee fit + filtering
#   --mask_ratio_spatial/temporal starting points, not tuned values
HARDENING="--random_mask --mask_ratio_spatial 0.3 --mask_ratio_temporal 0.3 \
    --mask_loss_on_masked_only --cache_bands --nan_policy raise"

# To reproduce the published pretraining exactly, set HARDENING="" and change
# --sequence_length_phase3 back to whatever the original run truly used (348).

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
    --sequence_length_phase3 464 \
    --nEpochs_phase3 ${N_EPOCHS} \
    --batch_size_phase3 4 \
    --lr_init_phase3 3e-5 \
    --num_hub_ROIs 380 \
    --spatial_loss_factor ${SPATIAL_LOSS_FACTOR} \
    --seed ${SEED} \
    --exp_name ${EXP_NAME} \
    ${HARDENING}
