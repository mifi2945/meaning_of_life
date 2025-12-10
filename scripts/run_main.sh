#!/bin/bash

#SBATCH --job-name=ewarta    # Job name
#SBATCH --output=logs/run_main_%j.out    # Output file (%j will be replaced with the job ID)
#SBATCH --error=logs/run_main_%j.err    # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-2:0    # Time limit (DD-HH:MM)
#SBATCH --nodes=1    # Number of nodes
#SBATCH --ntasks=1    # Number of tasks
#SBATCH --cpus-per-task=8    # CPUs per task
#SBATCH --partition=teaching    # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here
#SBATCH --time=0-1:0
#SBATCH --gpus=4


# Set working directory to project root
# Use SLURM_SUBMIT_DIR if available (when running under SLURM), otherwise use script location
cd "$SLURM_SUBMIT_DIR/.."


# Load necessary modules (adjust based on your cluster setup)
# module load python/3.13
# module load cuda/11.8

# Activate your environment if using conda/venv
# source activate your_env_name
# or
# source /path/to/venv/bin/activate

# ==========================================
# Edit arguments below to customize the run
# ==========================================

# Run your code here
uv run python src/main.py \
    --search-type NS_Q \
    --steps 100 \
    --expansion -1 \
    --problem-type growth \

