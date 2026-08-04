export PYTHONPATH=.

echo "=== testing ==="
CUDA_VISIBLE_DEVICES=7 python test_main.py \
--checkpoint ./checkpoints/best_model.pth \
--test_cfg configs/testing.yaml \
--model_cfg configs/model.yaml
