# k-diffusion-uncond – Grayscale unconditional diffusion training

This repository wraps the original **[crowsonkb/k-diffusion]** code base in a simple set-up for **1-channel (black-and-white) unconditional generation**.

---

## Quick start (Linux/Mac bash)

```bash
git clone https://github.com/<your-user>/k-diffusion-uncond.git
cd k-diffusion-uncond

# 1- Install Python deps & clone k-diffusion
bash install.sh            # ⇠ one command, ~2–3 min on fast network

# 2- Prepare (or replace) the dataset
python scripts/prepare_dataset.py \
  --zip /path/to/vel_dataset_old.zip        # default is the author’s dataset

# 3- Train
python scripts/train_uncond.py \
  --data data/vel_dataset_processed \
  --config config/default_config.json \
  --name vel_model                           \
  --batch-size 4                             \
  --end-step 100_000

# 4- Sample
python scripts/sample_uncond.py \
  --ckpt checkpoints/vel_model_100000.safetensors \
  --config config/default_config.json \
  --out-dir samples --n 16
```

GPU & PyTorch build – `install.sh` defaults to CUDA 11.8 wheels.
For a different driver/runtime, edit the `pip install torch torchvision ...` line accordingly.

### Arguments you will touch most often

| flag (train_uncond.py) | default | meaning |
| ---------------------- | ------- | ------- |
| --data | data/vel_dataset_processed | root folder of 1-channel PNGs |
| --batch-size | 4 | per-step micro-batch (effective batch = batch × grad_acc_steps) |
| --mixed-precision | bf16 | set to fp16 or leave blank for FP32 |
| --end-step | None | stop after N optimisation steps |

### Changing the dataset

ZIP file → `prepare_dataset.py --zip my.zip --out data/my_processed`

Already a folder of PNGs → `prepare_dataset.py --src my_pngs/`

After that, point `--data` at the processed directory.

### File descriptions

| path | role |
| ---- | ---- |
| install.sh | upgrades pip, installs pinned deps, and clones k-diffusion |
| requirements.txt | non-PyTorch packages (accelerate, kornia, …) |
| scripts/prepare_dataset.py | Converts any image type → 1-channel PNG, keeps directory structure |
| scripts/train_uncond.py | self-contained training loop using 🤗 Accelerate + EMA + gradient-checkpointing |
| scripts/sample_uncond.py | loads a .safetensors or .pth checkpoint and writes PNGs |
| config/default_config.json | minimal UNet-like model (no self-attention) tuned for 256×256 BW images |

© 2025 Ege Çırakman – MIT License

## 3. `install.sh`

```bash
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
```
