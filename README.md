# k-diffusion-uncond – Grayscale unconditional diffusion training

This repository wraps the original **[crowsonkb/k-diffusion]** code base in a
simple set-up for **1-channel (black-and-white) unconditional generation**.

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
