#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src"

TRAIN=${TRAIN:-/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train}
TEST=${TEST:-/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test}
RUN=${RUN:-RoboNanoVLM_results}
CKPT=${CKPT:-../checkpoints/}

echo "=== blind ablation ==="
CUDA_VISIBLE_DEVICES=7 python blind_ablation.py \
    --checkpoint "$CKPT" \
    --out ../logs/$RUN/blind_ablation