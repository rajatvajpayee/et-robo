#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src"

TEST=${TEST:-/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test}
RUN=${RUN:-RoboNanoVLM_results}
CKPT=${CKPT:-../checkpoints/}

echo "=== saliency ==="
python saliency.py \
    --test-path "$TEST" \
    --checkpoint "$CKPT" \
    --out ../logs/$RUN/saliency