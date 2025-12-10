#!/bin/bash

#SBATCH --job-name=ewarta    # Job name
#SBATCH --output=logs/nsq_parallel_%j.out    # Output file (%j will be replaced with the job ID)
#SBATCH --error=logs/nsq_parallel_%j.err    # Error file (%j will be replaced with the job ID)
#SBATCH --time=0-2:0    # Time limit (DD-HH:MM)
#SBATCH --nodes=1    # Number of nodes
#SBATCH --ntasks=1    # Number of tasks
#SBATCH --cpus-per-task=8    # CPUs per task
#SBATCH --mem=16G    # Memory per node
#SBATCH --gres=gpu:1    # GPU resources
#SBATCH --partition=teaching    # Partition to submit to. `teaching` (for the T4 GPUs) is default on Rosie, but it's still being specified here

# Set working directory to project root first
cd $SLURM_SUBMIT_DIR/..

# Create logs directory if it doesn't exist (must be after cd)
mkdir -p logs
mkdir -p results

# Load necessary modules (adjust based on your cluster setup)
# module load python/3.13
# module load cuda/11.8

# Activate your environment if using conda/venv
# source activate your_env_name
# or
# source /path/to/venv/bin/activate

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CUDA Available: $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA Device: $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# Test parameters (can be overridden with environment variables)
RUNS=${RUNS:-5}
STEPS=${STEPS:-100}
POP_SIZE=${POP_SIZE:-100}
NUM_EPOCHS=${NUM_EPOCHS:-50}
NO_COMPARE=${NO_COMPARE:-0}  # Set to 1 to skip sequential comparison

echo "=========================================="
echo "Testing NS-Q Parallel Performance"
echo "=========================================="
echo "Runs: $RUNS"
echo "Steps: $STEPS"
echo "Population Size: $POP_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Compare Sequential: $([ $NO_COMPARE -eq 1 ] && echo 'No' || echo 'Yes')"
echo "=========================================="

# Run your code here
uv run python src/test_nsq_performance.py \
    --runs $RUNS \
    --steps $STEPS \
    --pop-size $POP_SIZE \
    --epochs $NUM_EPOCHS \
    $([ $NO_COMPARE -eq 1 ] && echo "--no-compare") \
    2>&1 | tee logs/nsq_performance_test_${SLURM_JOB_ID}.log

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

