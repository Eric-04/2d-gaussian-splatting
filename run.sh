#!/bin/bash

#SBATCH -J surfel_splatting
#SBATCH -n 4
#SBATCH --mem=80G
#SBATCH -t 20:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C ampere
#SBATCH -o logs/surfel_%j_%x.out
#SBATCH -e logs/surfel_%j_%x.err

# Usage:
#   sbatch run.sh train
#   sbatch run.sh render

# ── Modules ──────────────────────────────────────────────────────────────────
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
module load cuda/11.8

# ── Environment ──────────────────────────────────────────────────────────────
conda activate surfel_splatting

REPO=/users/ekcho/repo/2d-gaussian-splatting
cd $REPO

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Starting at $(date)"

# Train
# python train.py -s data/360_v2/garden -m output/m360/garden

# Render
python render.py -m $REPO/output/m360/garden -s data/360_v2/garden \
    --render_path --skip_train --skip_test --skip_mesh

echo "Done at $(date)"