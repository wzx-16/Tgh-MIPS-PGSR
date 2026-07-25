"""Per-view PSNR comparison of train renders vs dataset GT for runs 61 and 66.

Uses manifest.tsv (file -> image_name) to fetch GT from the source dataset.
Prints per-camera-frame PSNR for both runs plus aggregate stats.
"""
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

DATA = Path("/root/autodl-tmp/projects/Tgh-MIPS-PGSR/data/N3V/abuzabi/images")
OUT = Path("/root/autodl-tmp/projects/4dRefgs_abuzabi/output/abuzabi")
to_tensor = transforms.ToTensor()

def load(p):
    return to_tensor(Image.open(p).convert("RGB")).cuda()

def psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    return float(-10.0 * torch.log10(mse))

def run_metrics(run_id):
    base = OUT / f"train_{run_id}" / "ours_80000"
    manifest = {}
    with open(base / "manifest.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            manifest[row["file"]] = row["image_name"]
    scores = {}
    for fname, image_name in sorted(manifest.items()):
        render_path = base / "renders" / fname
        gt_path = DATA / f"{image_name}.jpg"
        if not render_path.exists() or not gt_path.exists():
            continue
        r = load(render_path)
        g = load(gt_path)
        # train render files are (render | GT) composites, twice the GT width
        if r.shape[-1] == 2 * g.shape[-1]:
            r = r[..., : g.shape[-1]]
        if r.shape != g.shape:
            g = torch.nn.functional.interpolate(
                g.unsqueeze(0), size=r.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(0)
        scores[image_name] = psnr(r, g)
    return scores

s61 = run_metrics(61)
s66 = run_metrics(66)
common = sorted(set(s61) & set(s66))
print(f"common views: {len(common)}")
d = [s66[k] - s61[k] for k in common]
print(f"train PSNR  61: {sum(s61[k] for k in common)/len(common):.3f}")
print(f"train PSNR  66: {sum(s66[k] for k in common)/len(common):.3f}")
print(f"mean delta (66-61): {sum(d)/len(d):+.3f}")
worst = sorted(common, key=lambda k: s66[k] - s61[k])
print("\nlargest regressions (66-61):")
for k in worst[:8]:
    print(f"  {k}: 61={s61[k]:.2f} 66={s66[k]:.2f} d={s66[k]-s61[k]:+.2f}")
print("\nlargest gains (66-61):")
for k in worst[-8:]:
    print(f"  {k}: 61={s61[k]:.2f} 66={s66[k]:.2f} d={s66[k]-s61[k]:+.2f}")
