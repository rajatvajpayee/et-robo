#!/bin/bash

export PYTHONPATH=.

TEST_CFG="/home/rajat/scratch/et-robo/problem1_mnist/configs/testing.yaml"
MODEL_CFG_DIR="/home/rajat/scratch/et-robo/problem1_mnist/configs/model"

RESULT_DIR="results"
OUTPUT_DIR="runs"

mkdir -p ${RESULT_DIR}

###############################################
# Training Hyperparameter Experiments
# (Fixed model: model_baseline.yaml)
###############################################

BASELINE_MODEL_CFG="${MODEL_CFG_DIR}/model_baseline.yaml"

for CHECKPOINT in ${OUTPUT_DIR}/baseline ${OUTPUT_DIR}/small_batch ${OUTPUT_DIR}/medium_batch \
                  ${OUTPUT_DIR}/large_batch ${OUTPUT_DIR}/full_batch \
                  ${OUTPUT_DIR}/lower_lr ${OUTPUT_DIR}/moderate_lr \
                  ${OUTPUT_DIR}/higher_lr ${OUTPUT_DIR}/more_epochs
do
    CHECKPOINT="${CHECKPOINT}/best_model.pth"

    if [ ! -f "${CHECKPOINT}" ]; then
        continue
    fi

    RUN_NAME=$(basename "$(dirname "${CHECKPOINT}")")

    echo "Testing ${RUN_NAME}..."

    CUDA_VISIBLE_DEVICES=7 python test_main.py \
        --test_cfg "${TEST_CFG}" \
        --model_cfg "${BASELINE_MODEL_CFG}" \
        --checkpoint "${CHECKPOINT}" 
done


###############################################
# Model Architecture Experiments
# (Each model uses its own YAML)
###############################################

for MODEL_CFG in ${MODEL_CFG_DIR}/model_*.yaml
do
    RUN_NAME=$(basename "${MODEL_CFG}" .yaml)

    CHECKPOINT="${OUTPUT_DIR}/${RUN_NAME}/best_model.pth"

    if [ ! -f "${CHECKPOINT}" ]; then
        continue
    fi

    echo "Testing ${RUN_NAME}..."

    CUDA_VISIBLE_DEVICES=7 python test_main.py \
        --test_cfg "${TEST_CFG}" \
        --model_cfg "${MODEL_CFG}" \
        --checkpoint "${CHECKPOINT}" 
done
