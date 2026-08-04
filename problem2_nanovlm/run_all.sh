#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/src"

TRAIN=${TRAIN:-/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train}
TEST=${TEST:-/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test}
RUN=${RUN:-ft_with_head}

# echo "=== baseline (before fine-tuning) ==="
# python evaluator.py --train-path "$TRAIN" --test-path "$TEST" --out ../logs/baseline

echo "=== fine-tune ==="
python trainer.py --train-path "$TRAIN" --test-path "$TEST" --run-name "$RUN"

# CKPT=../logs/$RUN/ckpt/best

# echo "=== evaluate fine-tuned ==="
# python evaluator.py --train-path "$TRAIN" --test-path "$TEST" \
#   --checkpoint "$CKPT" --out ../logs/$RUN/eval

# echo "=== baseline vs fine-tuned ==="
# python compare.py --baseline ../logs/baseline --finetuned ../logs/$RUN/eval \
#   --out ../logs/$RUN/comparison

# echo "=== saliency ==="
# python saliency.py --test-path "$TEST" --checkpoint "$CKPT" --out ../logs/$RUN/saliency

# echo "=== decoding strategies ==="
# python sampling_study.py --train-path "$TRAIN" --test-path "$TEST" \
#   --checkpoint "$CKPT" --out ../logs/$RUN/sampling
