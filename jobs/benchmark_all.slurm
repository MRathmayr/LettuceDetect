#!/bin/bash
#SBATCH --job-name=bench-all
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --account=p73025
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/data/fs73025/mrathmayr/LettuceDetect/tests/benchmarks/results/%x_%j.out
#SBATCH --error=/gpfs/data/fs73025/mrathmayr/LettuceDetect/tests/benchmarks/results/%x_%j.err

# ============================================================
# FULL CASCADE BENCHMARK - Both Transformers x 4 PCA 512 Probes
# ============================================================
# Phase 1: Model-independent components (run once)
# Phase 2: Stage 3 standalone (once per LLM)
# Phase 3: Per-transformer benchmarks (base, large)
# Output: 8 JSON files (2 transformers x 4 probes)
# If a variant OOMs, Python skips it and continues.
# ============================================================

# Fail fast on setup errors, then allow Python to handle its own errors
set -e
export PYTHONUNBUFFERED=1
# MiniCheck-Flan-T5-Large uses legacy .tar checkpoint format,
# incompatible with PyTorch 2.6's weights_only=True default
export TORCH_FORCE_WEIGHTS_ONLY=0

echo "========================================"
echo "Full Cascade Benchmark - All Variants"
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
echo "Running full benchmark (2 transformers x 4 PCA 512 probes)..."
set +e
python tests/benchmarks/run_full_benchmark.py \
    --output tests/benchmarks/results/benchmark_all_halluprobe_predictions
BENCH_EXIT=$?
set -e

if [ $BENCH_EXIT -ne 0 ]; then
    echo "WARNING: Benchmark exited with code $BENCH_EXIT (partial results may be available)"
fi

echo ""
echo "========================================"
echo "Benchmark Complete - $(date)"
echo "========================================"
ls -lh "$LD_ROOT/tests/benchmarks/results/benchmark_all_halluprobe_predictions"/benchmark_*_*.json 2>/dev/null | tail -8
