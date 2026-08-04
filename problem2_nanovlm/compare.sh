#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src"

RUN=${RUN:-ft_with_head}

echo "=== baseline vs fine-tuned ==="
python compare.py \
    --baseline ../logs/baseline \
    --finetuned ../logs/$RUN/eval \
    --out ../logs/$RUN/comparison