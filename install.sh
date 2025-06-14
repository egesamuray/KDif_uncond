#!/usr/bin/env bash
set -e

echo "🔧  Setting up Python environment …"
python3 -m pip install --upgrade pip

# ---- core: torch & torchvision ----------------------------------------------
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# ---- rest of the deps -------------------------------------------------------
pip install -r requirements.txt

# ---- clone k-diffusion (fixed commit) ---------------------------------------
if [ ! -d "k-diffusion" ]; then
  echo "⬇️  Cloning k-diffusion …"
  git clone --depth 1 https://github.com/crowsonkb/k-diffusion.git
fi

echo "✅  Environment ready."
