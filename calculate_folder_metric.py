import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]


def list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def match_pairs(render_files: List[Path], gt_files: List[Path]) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    gt_by_name = {p.name.lower(): p for p in gt_files}
    gt_by_stem = {}
    for p in gt_files:
        # Keep first by sorted order when duplicated stems exist.
        gt_by_stem.setdefault(p.stem.lower(), p)

    pairs = []
    missing = []
    for render_path in render_files:
        gt_path = gt_by_name.get(render_path.name.lower())
        if gt_path is None:
            gt_path = gt_by_stem.get(render_path.stem.lower())
        if gt_path is None:
            missing.append(render_path)
            continue
        pairs.append((render_path, gt_path))
    return pairs, missing


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2)
    if mse.item() <= 1e-12:
        return float("inf")
    return float((-10.0 * torch.log10(mse)).item())


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    xs = torch.arange(window_size, dtype=torch.float32)
    gauss = torch.exp(-((xs - window_size // 2) ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def _create_ssim_window(window_size: int, channel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    window_1d = _gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device=device, dtype=dtype)


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    channel = pred.size(1)
    window = _create_ssim_window(window_size, channel, pred.dtype, pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return float(ssim_map.mean().item())


def build_lpips_model(device: torch.device):
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="alex").to(device).eval()
        return model, "lpips"
    except Exception:
        pass

    try:
        from lpipsPyTorch.modules.lpips import LPIPS as LocalLPIPS

        model = LocalLPIPS(net_type="alex").to(device).eval()
        return model, "lpipsPyTorch"
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize LPIPS(alex). Install the 'lpips' package or ensure lpipsPyTorch can load weights."
        ) from exc


def compute_lpips(pred: torch.Tensor, gt: torch.Tensor, lpips_model, lpips_backend: str) -> float:
    if lpips_backend == "lpips":
        value = lpips_model(pred, gt, normalize=True)
    else:
        value = lpips_model(pred, gt)
    return float(value.mean().item())


def resolve_folders(args: argparse.Namespace) -> Tuple[Path, Path]:
    # New argument names.
    renders = args.renders
    gt = args.gt

    # Backward compatibility with old script args.
    if renders is None:
        renders = args.folder2
    if gt is None:
        gt = args.folder1

    if renders is None or gt is None:
        raise ValueError("Please provide --renders and --gt (or legacy --folder2 and --folder1).")

    return Path(renders), Path(gt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate renders against GT with PSNR, SSIM, and LPIPS(alex)."
    )
    parser.add_argument("--renders", type=str, default=None, help="Path to renders folder.")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground-truth folder.")

    # Backward-compatible aliases from the old script.
    parser.add_argument("--folder1", type=str, default=None, help="Legacy alias for --gt.")
    parser.add_argument("--folder2", type=str, default=None, help="Legacy alias for --renders.")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for metric computation.")
    parser.add_argument("--resize", action="store_true", help="Resize render image to GT resolution when sizes mismatch.")
    parser.add_argument("--max-images", type=int, default=0, help="Evaluate at most this many images (0 means all).")
    parser.add_argument("--per-image", action="store_true", help="Print per-image metrics.")
    args = parser.parse_args()

    try:
        renders_dir, gt_dir = resolve_folders(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    render_files = list_images(renders_dir)
    gt_files = list_images(gt_dir)
    if not render_files:
        raise RuntimeError(f"No images found in renders folder: {renders_dir}")
    if not gt_files:
        raise RuntimeError(f"No images found in gt folder: {gt_dir}")

    pairs, missing = match_pairs(render_files, gt_files)
    if args.max_images > 0:
        pairs = pairs[:args.max_images]

    if not pairs:
        raise RuntimeError("No matched image pairs found between renders and gt folders.")

    if missing:
        print(f"Warning: {len(missing)} render images had no matching GT and were skipped.")

    lpips_model, lpips_backend = build_lpips_model(device)
    print(f"LPIPS backend: {lpips_backend} (alex)")

    psnr_values: List[float] = []
    ssim_values: List[float] = []
    lpips_values: List[float] = []
    skipped_size_mismatch = 0

    with torch.no_grad():
        for render_path, gt_path in tqdm(pairs, desc="Evaluating"):
            pred = read_image(render_path).to(device)
            gt = read_image(gt_path).to(device)

            if pred.shape != gt.shape:
                if args.resize:
                    pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    skipped_size_mismatch += 1
                    continue

            pred = pred.clamp(0.0, 1.0)
            gt = gt.clamp(0.0, 1.0)

            psnr_val = compute_psnr(pred, gt)
            ssim_val = compute_ssim(pred, gt)
            lpips_val = compute_lpips(pred, gt, lpips_model, lpips_backend)

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            lpips_values.append(lpips_val)

            if args.per_image:
                print(
                    f"{render_path.name}: "
                    f"PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, LPIPS(alex)={lpips_val:.4f}"
                )

    evaluated = len(psnr_values)
    if evaluated == 0:
        raise RuntimeError(
            "No image pairs were evaluated. Use --resize if renders and gt have different resolutions."
        )

    avg_psnr = sum(psnr_values) / evaluated
    avg_ssim = sum(ssim_values) / evaluated
    avg_lpips = sum(lpips_values) / evaluated

    print("\n=== Folder Metrics ===")
    print(f"Renders: {renders_dir}")
    print(f"GT: {gt_dir}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Evaluated pairs: {evaluated}")
    if skipped_size_mismatch > 0:
        print(f"Skipped (size mismatch): {skipped_size_mismatch} (use --resize to include them)")

    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"LPIPS(alex): {avg_lpips:.4f}")


if __name__ == "__main__":
    main()
