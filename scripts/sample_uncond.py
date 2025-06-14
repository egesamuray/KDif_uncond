#!/usr/bin/env python3
"""Generate PNG samples from a trained checkpoint."""

import argparse, json, math
from pathlib import Path
import torch, numpy as np
import k_diffusion as K
from torchvision.utils import make_grid

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help=".safetensors or .pth")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("samples"))
    p.add_argument("--n", type=int, default=16, help="#samples")
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    cfg       = json.loads(args.config.read_text())
    model_cfg = cfg["model"]
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    model = K.config.make_model(cfg).to(device).eval()
    if args.ckpt.suffix == ".safetensors":
        import safetensors.torch as safetorch
        model.load_state_dict(safetorch.load_file(args.ckpt))
    else:
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck.get("model_ema", ck))

    den = K.Denoiser(model, sigma_data=model_cfg["sigma_data"])
    sigmas = K.sampling.get_sigmas_karras(args.steps,
                                          model_cfg["sigma_min"],
                                          model_cfg["sigma_max"],
                                          rho=7.0, device=device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        torch.manual_seed(i + 1000)
        x = torch.randn(1, 1, *model_cfg["input_size"], device=device) * sigmas.max()
        x0 = K.sampling.sample_dpmpp_2m_sde(den, x, sigmas, eta=0.0, solver_type="heun")
        fn = args.out_dir / f"sample_{i:05}.png"
        K.utils.to_pil_image(x0[0]).save(fn)
    print("✅  wrote", args.n, "images to", args.out_dir)

if __name__ == "__main__":
    main()
