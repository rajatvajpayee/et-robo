#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src"

TRAIN=${TRAIN:-/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train}
TEST=${TEST:-/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test}
RUN=${RUN:-RoboNanoVLM_FT}

echo "=== fine-tune ==="
CUDA_VISIBLE_DEVICES=7 python trainer.py \
    --train-path "$TRAIN" \
    --test-path "$TEST" \
    --run-name ../checkpoints/"$RUN" 