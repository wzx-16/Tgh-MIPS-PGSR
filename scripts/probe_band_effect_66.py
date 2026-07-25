"""Probe run-66 hierarchical residual bands (concise stats)."""
import sys
sys.path.insert(0, "/root/autodl-tmp/projects/4dRefgs_abuzabi")
import torch

enc = torch.load(
    "output/abuzabi/dir66/iteration_80000/dir_encoding.pt",
    map_location="cuda", weights_only=False,
)
enc.eval()

H, W = 64, 128
ys = torch.linspace(0.02, 0.98, H, device="cuda")
xs = torch.linspace(0.02, 0.98, W, device="cuda")
grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)
wo_xyz = grid[None, :, None, :]
level = torch.full((grid.shape[0], 1), 0.15, device="cuda")

T0, T1 = 0.6667, 2.6333

def query(t):
    with torch.no_grad():
        out = enc(wo_xyz.clone(), level, index=0, timestamp=t, iteration=None)
        return out.reshape(-1, 16)

base = query(None)  # timestamp None -> residuals skipped
print(f"static base: mean|f| = {base.abs().mean():.4f}")
print("t      |full-base|  rel%")
for t in torch.linspace(T0, T1, 9):
    full = query(float(t))
    eff = (full - base).abs().mean()
    print(f"{float(t):.3f}  {eff:.4f}      {100 * eff / base.abs().mean():.1f}%")

feats = torch.stack([query(float(t)) for t in torch.linspace(T0, T1, 61)])
std_t = feats.std(dim=0).mean(dim=0)
print("\nper-channel temporal std (ch0-7 static, ch8-11 band0, ch12-14 band1, ch15 band2):")
print("  " + " ".join(f"{v:.4f}" for v in std_t.tolist()))
print(f"mean feature magnitude: {feats.abs().mean():.4f}")

step = (feats[1:] - feats[:-1]).abs().mean(dim=(1, 2))
print(f"\nframe-to-frame |delta| mean={step.mean():.5f} max={step.max():.5f} min={step.min():.5f}")
