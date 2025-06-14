#!/usr/bin/env python3
"""Memory-optimised unconditional training for grayscale images.

* Requires: k-diffusion clone under repo root.
* Uses HuggingFace Accelerate for multi-GPU & mixed precision.
"""

import argparse, json, os, subprocess, time
from copy import deepcopy
from pathlib import Path

import torch, accelerate
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

REPO_DIR = Path(__file__).resolve().parents[1]
KD_PATH  = REPO_DIR / "k-diffusion"

# ----------------------------------------------------------------------------- #
# utils
# ----------------------------------------------------------------------------- #
def ensure_kdiff():
    if KD_PATH.exists():
        return
    print("⬇️  Cloning k-diffusion …")
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/crowsonkb/k-diffusion.git",
                    str(KD_PATH)], check=True)

class GrayDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted([p for p in Path(root).rglob("*.png")])
        if not self.paths:
            raise RuntimeError(f"No PNGs found in {root}")
        print(f"Dataset size: {len(self.paths)}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            if im.mode != "L":
                im = im.convert("L")
            arr = np.array(im, dtype=np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).unsqueeze(0)   # 1×H×W
        aug_cond = torch.zeros(9)                         # dummy
        return tensor, 0, aug_cond

def main():
    ensure_kdiff()
    import k_diffusion as K                               # noqa: E402
    from torch import optim                               # noqa: E402

    # ---- CLI ------------------------------------------------------------------
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/vel_dataset_processed"))
    p.add_argument("--config", type=Path, default=Path("config/default_config.json"))
    p.add_argument("--name", type=str, default="model")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--mixed-precision", choices=["bf16","fp16"], default="bf16")
    p.add_argument("--checkpoint", action="store_true", help="enable gradient ckpt.")
    p.add_argument("--end-step", type=int, default=None)
    args = p.parse_args()

    # ---- Accelerator ----------------------------------------------------------
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision=args.mixed_precision
    )
    device = accelerator.device

    # ---- Config & model -------------------------------------------------------
    cfg = json.loads(Path(args.config).read_text())
    model_cfg = cfg["model"]
    assert model_cfg["input_channels"] == 1, "config must be 1-channel"

    model      = K.config.make_model(cfg)
    model_ema  = deepcopy(model)
    opt        = optim.AdamW(model.parameters(), **cfg["optimizer"])
    sched      = K.config.make_lr_sched(cfg)(opt)
    ema_sched  = K.utils.EMAWarmup(**cfg["ema_sched"])

    train_ds   = GrayDataset(args.data)
    train_dl   = DataLoader(train_ds, args.batch_size,
                            shuffle=True, drop_last=True,
                            num_workers=args.num_workers, pin_memory=True)
    model, model_ema, opt, train_dl = accelerator.prepare(model, model_ema, opt, train_dl)

    denoiser, denoiser_ema = (K.config.make_denoiser_wrapper(cfg)(m)
                              for m in (model, model_ema))
    sample_density = K.config.make_sample_density(model_cfg)
    sigma_min, sigma_max = model_cfg["sigma_min"], model_cfg["sigma_max"]

    # ---- training -------------------------------------------------------------
    step, losses = 0, []
    try:
        while True:
            for reals, _, aug in tqdm(train_dl, disable=not accelerator.is_local_main_process):
                with accelerator.accumulate(model):
                    noise  = torch.randn_like(reals)
                    sigma  = sample_density([reals.size(0)], device=device)
                    with K.models.checkpointing(args.checkpoint):
                        loss_val = denoiser.loss(reals, noise, sigma, aug_cond=aug)
                    accelerator.backward(loss_val.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step(); sched.step(); opt.zero_grad()
                        K.utils.ema_update(model, model_ema, ema_sched.get_value())
                        ema_sched.step()

                losses.append(loss_val.detach().mean().item())
                if accelerator.sync_gradients:
                    if step % 50 == 0 and accelerator.is_main_process:
                        print(f"step {step}  loss {np.mean(losses):.4f}")
                        losses.clear()
                    step += 1
                    if args.end_step and step >= args.end_step:
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            ck = f"checkpoints/{args.name}_{step:08}.safetensors"
            os.makedirs("checkpoints", exist_ok=True)
            accelerator.save(accelerator.unwrap_model(model_ema).state_dict(), ck)
            print("✅  saved", ck)

if __name__ == "__main__":
    main()
