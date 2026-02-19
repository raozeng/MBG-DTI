#!/bin/bash

# Comparative Experiments Runner
# This script runs the proposed Mamba-BiLSTM model and 3 baselines.
# It uses the Davis dataset by default.

# Common Settings
DATA_PATH="./data/Davis.txt"
DATASET_NAME="Davis"
EPOCHS=100
BATCH_SIZE=64
FOLDS=5

echo "========================================================"
echo "Starting Comparative Experiments on $DATASET_NAME"
echo "========================================================"

# 1. Proposed Model: Mamba-BiLSTM
echo ""
echo "[1/4] Running Proposed Model: Mamba-BiLSTM"
echo "--------------------------------------------------------"
python run.py train \
    --model_name mamba_bilstm \
    --data $DATA_PATH \
    --dataset_name $DATASET_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --folds $FOLDS \
    --fine_tune

# 2. Baseline 1: DeepDTA (CNN-based)
echo ""
echo "[2/4] Running Baseline: DeepDTA (CNN)"
echo "--------------------------------------------------------"
python run.py train \
    --model_name deepdta \
    --data $DATA_PATH \
    --dataset_name $DATASET_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --folds $FOLDS

# 3. Baseline 2: TransformerDTI (Transformer-based, MolTrans-inspired)
echo ""
echo "[3/4] Running Baseline: TransformerDTI"
echo "--------------------------------------------------------"
python run.py train \
    --model_name transformerdti \
    --data $DATA_PATH \
    --dataset_name $DATASET_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --folds $FOLDS

# 4. Baseline 3: MCANet (Attention-based)
echo ""
echo "[4/4] Running Baseline: MCANet"
echo "--------------------------------------------------------"
python run.py train \
    --model_name mcanet \
    --data $DATA_PATH \
    --dataset_name $DATASET_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --folds $FOLDS

echo ""
echo "========================================================"
echo "All Experiments Completed."
echo "Results are saved in $DATASET_NAME/<model_name>/train_result/"
echo "========================================================"
