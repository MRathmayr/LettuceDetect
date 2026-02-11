#!/bin/bash
#SBATCH --job-name=bench-val
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --account=p73025
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/data/fs73025/mrathmayr/LettuceDetect/tests/benchmarks/results/%x_%j.out
#SBATCH --error=/gpfs/data/fs73025/mrathmayr/LettuceDetect/tests/benchmarks/results/%x_%j.err

# ============================================================
# VALIDATION RUN - Verify 3-phase pipeline end-to-end
# ============================================================
# 10 samples, both transformers (base + large), 3B probe only
# Tests: Phase 1 + Phase 2 (3B standalone) + Phase 3 (base+3B, large+3B)
# Output: benchmark_base_3b_*.json, benchmark_large_3b_*.json
# ============================================================

set -e
export PYTHONUNBUFFERED=1
# MiniCheck-Flan-T5-Large uses legacy .tar checkpoint format,
# incompatible with PyTorch 2.6's weights_only=True default
export TORCH_FORCE_WEIGHTS_ONLY=0

echo "========================================"
echo "Benchmark Validation Run (10 samples, 3B, both transformers)"
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
export PYTHONPATH="$LD_ROOT:$PYTHONPATH"

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p "$LD_ROOT/tests/benchmarks/results"

echo ""
echo "Running validation (10 samples, 3B probe, both transformers)..."
python tests/benchmarks/run_full_benchmark.py \
    --limit 10 \
    --stage3 3b

echo ""
echo "========================================"
echo "Validation Complete - $(date)"
echo "========================================"
ls -lh "$LD_ROOT/tests/benchmarks/results"/benchmark_*_3b_*.json 2>/dev/null | tail -4
