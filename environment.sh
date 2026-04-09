#!/bin/bash

conda create -n mbbn python=3.10 -y
conda activate mbbn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu114
pip install nibabel nilearn nitime timm tensorboard numpy pandas wandb weightwatcher tqdm scikit-learn scikit-image matplotlib transformers lmfit