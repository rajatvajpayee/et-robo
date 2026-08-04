#!/bin/bash

export PYTHONPATH=.

TRAIN_CFG="/home/rajat/scratch/et-robo/problem1_mnist/configs/training/baseline.yaml"
MODEL_CFG_DIR="/home/rajat/scratch/et-robo/problem1_mnist/configs/model"

mkdir -p logs

for MODEL_CFG in ${MODEL_CFG_DIR}/*.yaml
do
    RUN_TITLE=$(basename "${MODEL_CFG}" .yaml)

    echo "Launching ${RUN_TITLE}..."

    CUDA_VISIBLE_DEVICES=7 python train_main.py \
        --run_title "${RUN_TITLE}" \
        --train_cfg "${TRAIN_CFG}" \
        --model_cfg "${MODEL_CFG}" \
        > logs/${RUN_TITLE}.log 2>&1 &
done

echo "All model experiments launched."

wait