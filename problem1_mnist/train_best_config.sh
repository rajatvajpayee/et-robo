export PYTHONPATH=.

echo "=== training ==="
CUDA_VISIBLE_DEVICES=7 python train_main.py \
--run_title fnf_baseline_model \
--train_cfg configs/training.yaml \
--model_cfg configs/model.yaml
