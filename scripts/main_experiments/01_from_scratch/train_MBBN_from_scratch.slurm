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

# ── Hardening (docs/audit/AUDIT.md) ──────────────────────────────────────────
#   --head_type linear        plain read-out. The published head batch-norms and
#                             then drops out the *scalar logit*, which makes one
#                             subject's prediction depend on its batch-mates
#                             (SD 0.29 at batch 16) and zeroes 61% of logits.
#   --spat_diff_loss_type     `minus_log` (published) is unbounded, singular
#                             where the bands agree, and its global optimum is
#                             one-hot (hub) attention. `neg_linear` is bounded.
#   --spatial_loss_warmup     ramp λ so the 1/S transient at init cannot dominate
#   --spatial_head            put the spatial maps on the prediction path; without
#                             it d(prediction)/d(spatial attention) is exactly 0
#   --band_embedding          let the shared encoder tell the three bands apart
#   --site_stratify           balance acquisition site across folds
#   --group_by_family         keep ABCD siblings inside one fold
#   --cache_bands             memoise the per-subject knee fit + filtering
HARDENING="--head_type linear --spat_diff_loss_type neg_linear \
    --spatial_loss_warmup 500 --spatial_head --band_embedding \
    --site_stratify --cache_bands --nan_policy raise --split_seed 1"
# ABCD only: add --group_by_family to HARDENING.
# To reproduce the published run exactly:
#   HARDENING="--head_type published --spat_diff_loss_type minus_log \
#              --eval_test_every_epoch --keep_all_best_checkpoints --no_padding_mask" 

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
    --exp_name ${EXP_NAME} \
    ${HARDENING}
