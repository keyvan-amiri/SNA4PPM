#!/bin/bash

#SBATCH --job-name=gpu_models
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=24:00:00

# Define the Python script to run
PYTHON_SCRIPT="run_tabular_experiment.py"

# Define models and datasets
# MODELS=("TabM" "MLP" "RealMLP" "XGBoost" "CatBoost")
MODELS=("TabM" "CatBoost")
DATASETS=('BPIC15_5' 'BPIC20PTC' 'BPIC15_3' 'HelpDesk' 'BPIC20DD' 'BPIC15_1' 'BPIC15_2' 'BPIC15_4' 'BPIC20ID' 'BPIC17')
FE_TYPES=("None")

# Optional arguments
GPUS="0"
SEEDS_PARALLEL=5
# FE_TYPE="None"
FE_ORDER=0
N_TRIALS=0
SAVE_INTERVAL=1
VAL_STRATEGY="5CV-random"

# Loop through each model and dataset combination
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for FE_TYPE in "${FE_TYPES[@]}"; do
            echo "Running: python $PYTHON_SCRIPT $model $dataset --gpus $GPUS --seeds_parallel $SEEDS_PARALLEL --fe_type $FE_TYPE --fe_order $FE_ORDER --n_trials $N_TRIALS --save_interval $SAVE_INTERVAL --val_strategy $VAL_STRATEGY"
            python "$PYTHON_SCRIPT" "$model" "$dataset" --gpus "$GPUS" --seeds_parallel "$SEEDS_PARALLEL" --fe_type "$FE_TYPE" --fe_order "$FE_ORDER" --n_trials "$N_TRIALS" --save_interval "$SAVE_INTERVAL" --val_strategy "$VAL_STRATEGY"
            echo "------------------------------"
        done
    done
done