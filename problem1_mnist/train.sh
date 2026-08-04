#!/bin/bash

export PYTHONPATH=.

MODEL_CFG="/home/rajat/scratch/et-robo/problem1_mnist/configs/model/model_baseline.yaml"
TRAIN_CFG_DIR="/home/rajat/scratch/et-robo/problem1_mnist/configs/stepsize"

mkdir -p logs

for TRAIN_CFG in ${TRAIN_CFG_DIR}/*.yaml
do
    RUN_TITLE=$(basename "${TRAIN_CFG}" .yaml)

    echo "Launching ${RUN_TITLE}..."

    CUDA_VISIBLE_DEVICES=7 python train_main.py \
        --run_title "${RUN_TITLE}" \
        --train_cfg "${TRAIN_CFG}" \
        --model_cfg "${MODEL_CFG}" \
        > logs/${RUN_TITLE}.log 2>&1 &
done

echo "All experiments launched."
wait