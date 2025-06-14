#!/usr/bin/env python3
"""Convert any folder/zip of images to 1-channel PNGs ready for training."""

import argparse, os, zipfile, shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def extract_zip(zpath: Path, out_dir: Path):
    print(f"Extracting {zpath} …")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(out_dir)

def to_grayscale(src: Path, dst: Path):
    with Image.open(src) as im:
        if im.mode != "L":
            im = im.convert("L")
        im.save(dst, "PNG")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path,
                   help="zip archive containing images / folders (optional)")
    p.add_argument("--src", type=Path,
                   help="folder containing images (alternative to --zip)")
    p.add_argument("--out", type=Path, default=Path("data/vel_dataset_processed"),
                   help="destination root")
    args = p.parse_args()

    root_tmp = Path("data/tmp_extract")
    if args.zip:
        root_tmp.mkdir(parents=True, exist_ok=True)
        extract_zip(args.zip, root_tmp)
        src_root = root_tmp
    elif args.src:
        src_root = args.src
    else:
        raise SystemExit("Either --zip or --src must be given.")

    # replicate folder structure
    img_paths = [p for p in src_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    print(f"Found {len(img_paths)} images")
    for pth in tqdm(img_paths):
        rel = pth.relative_to(src_root)
        dst = args.out / rel.with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            to_grayscale(pth, dst)
        except Exception as e:
            print("‼️", pth, e)

    if root_tmp.exists():
        shutil.rmtree(root_tmp)
    print(f"✅  Processed dataset saved to {args.out}")

if __name__ == "__main__":
    main()

