#!/bin/bash
# Run WeightWatcher analysis on a pretrained MBBN model.
# Results (CSV + PNG) are saved to --weightwatcher_save_dir.

cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

# ── User settings ────────────────────────────────────────────────────────────
PRETRAINED_WEIGHTS="/path/to/pretrained_model.pth"
SAVE_DIR="./weightwatcher/results"
INTERMEDIATE_VEC=360            # 360 (HCP-MMP1) | 400 (Schaefer)
EXP_NAME="weightwatcher_analysis"

# ── Run ──────────────────────────────────────────────────────────────────────
python main.py \
    --step 2 \
    --weightwatcher \
    --intermediate_vec ${INTERMEDIATE_VEC} \
    --pretrained_model_weights_path ${PRETRAINED_WEIGHTS} \
    --weightwatcher_save_dir ${SAVE_DIR} \
    --exp_name ${EXP_NAME} \
    --wandb_mode offline
