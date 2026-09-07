#!/bin/bash
#SBATCH --job-name=MBBN_interpretability
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%A_%x.out
#SBATCH --error=slurm_logs/%A_%x.err

# ── User settings ────────────────────────────────────────────────────────────
SEED=1

# Dataset / task
DATASET=ABCD                    # ABCD | UKB | ABIDE
TARGET=ADHD_label               # sex | ADHD_label | ASD
FINE_TUNE_TASK=binary_classification

# Atlas / architecture (must match the trained model)
#   HCP-MMP1 (360 ROIs): --intermediate_vec 360 --num_heads 12
#   Schaefer (400 ROIs): --intermediate_vec 400 --num_heads 8
INTERMEDIATE_VEC=360
NUM_HEADS=12

# Sequence length: ABCD=348, UKB=464, ABIDE=280
SEQ_LEN=348

# Spatial loss factor λ (must match the trained model)
SPATIAL_LOSS_FACTOR=10.0

# Path to the trained model checkpoint
MODEL_PATH="/path/to/trained_model.pth"

# Output directory for attribution maps
SAVE_DIR="./visualization/${DATASET}_${TARGET}_seed${SEED}"

# ── Attribution mode (docs/audit/AUDIT.md) ───────────────────────────────────
# `spatial_difference` reproduces the published maps: it backpropagates the
# band-repulsion loss, whose gradient is quantised to {-2,-1,0,1,2}/(numel*S)
# -- a handful of distinct values, 33% of them exactly zero -- and carries no
# label information. The diagnosis enters only via which subjects are averaged.
#
# `label_gradient` attributes the model's own decision variable, which requires
# the maps to be ON the prediction path: train with --spatial_head, otherwise
# d(prediction)/d(spatial attention) is exactly 0 and every value would be 0.
ATTRIBUTION=label_gradient       # label_gradient | spatial_difference
N_PERMUTATIONS=1000              # label-permutation null for group differences

# ── Environment ──────────────────────────────────────────────────────────────
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate mbbn
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

# ── Run ──────────────────────────────────────────────────────────────────────
python visualization.py \
    --dataset_name ${DATASET} \
    --fine_tune_task ${FINE_TUNE_TASK} \
    --target ${TARGET} \
    --intermediate_vec ${INTERMEDIATE_VEC} \
    --num_heads ${NUM_HEADS} \
    --sequence_length ${SEQ_LEN} \
    --spatial_loss_factor ${SPATIAL_LOSS_FACTOR} \
    --model_path ${MODEL_PATH} \
    --save_dir ${SAVE_DIR} \
    --seed ${SEED} \
    --attribution ${ATTRIBUTION} \
    --n_permutations ${N_PERMUTATIONS} \
    --spatial_head \
    --head_type linear \
    --confidence_margin 0.1 \
    --cache_bands \
    --wandb_mode disabled
