#!/bin/bash
# ============================================================
# LettuceDetect Setup for VSC-5
# ============================================================
# Installs LettuceDetect with cascade dependencies into a venv.
# Run this ONCE on a GPU node before submitting benchmark jobs.
#
# Usage (on interactive GPU node):
#   salloc --partition=zen3_0512_a100x2 --qos=zen3_0512_a100x2 \
#          --account=p73025 --gres=gpu:2 --time=00:30:00
#   bash $DATA/LettuceDetect/jobs/setup_vsc.sh
# ============================================================

set -e

export DATA="/gpfs/data/fs73025/mrathmayr"
export HF_HOME="$DATA/.cache/huggingface"
LD_ROOT="$DATA/LettuceDetect"
ENV_PATH="$LD_ROOT/venv"

echo "=============================================="
echo "LettuceDetect VSC Setup"
echo "=============================================="
echo "Root:        $LD_ROOT"
echo "Environment: $ENV_PATH"
echo ""

# Check if venv already exists
if [ -d "$ENV_PATH" ]; then
    echo "Venv already exists - verifying..."
    source "$ENV_PATH/bin/activate"
    python -c "import lettucedetect; print(f'LettuceDetect installed')"
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print(f'spaCy: {spacy.__version__}, model OK')"
    echo ""
    echo "Setup OK - skipping creation"
    exit 0
fi

# Need GPU node for CUDA-aware packages
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found!"
    echo "First-time setup must run on a GPU compute node."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Create venv
echo "Loading Python via miniforge3..."
module purge
module load miniforge3
eval "$(conda shell.bash hook)"

echo "Creating venv at $ENV_PATH..."
python -m venv --upgrade-deps "$ENV_PATH"
source "$ENV_PATH/bin/activate"
pip install --upgrade pip

# Install LettuceDetect with all cascade dependencies
echo ""
echo "Installing LettuceDetect[cascade]..."
cd "$LD_ROOT"
pip install -e ".[cascade,dev]"

# Download spaCy model
echo ""
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data (needed by lexical overlap)
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# Verify
echo ""
echo "=============================================="
echo "Verifying installation"
echo "=============================================="
python -c "import lettucedetect; print('LettuceDetect: OK')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print(f'spaCy: {spacy.__version__}, model OK')"
python -c "import model2vec; print(f'Model2Vec: OK')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
python -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
