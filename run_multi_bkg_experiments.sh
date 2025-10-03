#!/bin/bash

# Script to run multiple training experiments with different numbers of backgrounds
# Usage: bash run_multi_bkg_experiments.sh

set -e  # Exit on any error

# Configuration
DATA_FILE="all_data.mat"
EPOCHS=50
BATCH_SIZE=8
LR=5e-4
BASE_CHANNELS=64
VAL_SPLIT=0.05
NUM_WORKERS=2
SEED=42

# Array of background counts to test
BKG_COUNTS=(1 5 25 50)

# Base experiment name
BASE_EXP_NAME="mbg_train_synth_test_cal_exp"

echo "========================================"
echo "Multi-Background Training Experiments"
echo "========================================"
echo "Data file: $DATA_FILE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Background counts: ${BKG_COUNTS[@]}"
echo "========================================"
echo ""

# Loop through each background count and run training
for NUM_BKGS in "${BKG_COUNTS[@]}"; do
    EXP_TAG="${BASE_EXP_NAME}_${NUM_BKGS}bkgs"
    
    echo "========================================"
    echo "Starting experiment: $EXP_TAG"
    echo "Number of backgrounds: $NUM_BKGS"
    echo "========================================"
    
    # Run training
    uv run train.py \
        --data "$DATA_FILE" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --base-channels $BASE_CHANNELS \
        --val-split $VAL_SPLIT \
        --num-workers $NUM_WORKERS \
        --seed $SEED \
        --num-backgrounds $NUM_BKGS \
        --experiment-tag "$EXP_TAG"
    
    echo ""
    echo "Completed experiment: $EXP_TAG"
    echo ""
    
    # Optional: Add a small delay between experiments
    sleep 2
done

echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Lightning logs: lightning_logs/${BASE_EXP_NAME}_*bkgs/"
echo "  - Training figures: figures/${BASE_EXP_NAME}_*bkgs/"
echo ""

