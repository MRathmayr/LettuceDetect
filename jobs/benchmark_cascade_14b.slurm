#!/bin/bash
#SBATCH --job-name=bench-cascade-14b
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --account=p73025
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/data/fs73025/mrathmayr/LettuceDetect/tests/benchmarks/results/%x_%j.out
#SBATCH --error=/gpfs/data/fs73025/mrathmayr/LettuceDetect/tests/benchmarks/results/%x_%j.err

# ============================================================
# FULL CASCADE BENCHMARK - Qwen 2.5 14B
# ============================================================
# Runs on feature/task-routing which benchmarks BOTH:
#   - Plain Cascade [1,3] (pre-refactor baseline)
#   - Task-Routed Cascade [1,3] (QA -> [1,3], other -> [1])
# ============================================================

set -e
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Full Cascade Benchmark - Qwen 2.5 14B"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

export DATA="/gpfs/data/fs73025/mrathmayr"
export HF_HOME="$DATA/.cache/huggingface"
LD_ROOT="$DATA/LettuceDetect"

source "$LD_ROOT/venv/bin/activate"
cd "$LD_ROOT"

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p "$LD_ROOT/tests/benchmarks/results"

echo ""
echo "Running full benchmark (--all-datasets --stage3 14b)..."
python tests/benchmarks/run_full_benchmark.py \
    --all-datasets \
    --stage3 14b

echo ""
echo "========================================"
echo "Benchmark Complete - $(date)"
echo "========================================"
ls -lh "$LD_ROOT/tests/benchmarks/results"/benchmark_14b_*.json 2>/dev/null | tail -2
