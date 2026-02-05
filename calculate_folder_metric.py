import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import math

def read_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = Image.open(path).convert('RGB')
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # 1, C, H, W

def read_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = Image.open(path).convert('L') # Read as grayscale
    transform = transforms.ToTensor()
    mask = transform(img).unsqueeze(0) # 1, 1, H, W
    # render.py logic is > 0. If saved as image, 1/255 is approx 0.0039.
    # We use a small threshold to capture any non-zero pixels.
    mask = (mask > 0.0).float()
    return mask

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_masked(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Ensure mask has same channels as image for proper broadcasting/counting
    if mask.shape[1] == 1 and img1.shape[1] == 3:
        mask = mask.repeat(1, 3, 1, 1)

    mse = (((img1 - img2) * mask)) ** 2
    # Sum of squared errors / (Count of masked pixels)
    # Add epsilon to denominator to avoid division by zero
    mse = mse.view(img1.shape[0], -1).sum(1, keepdim=True) / (mask.view(img1.shape[0], -1).sum(1, keepdim=True) + 1e-8)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    parser = argparse.ArgumentParser(description="Calculate average PSNR between two sets of images, optionally using a mask.")
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first folder (e.g., GT).")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the second folder (e.g., Render).")
    #parser.add_argument("--mask_folder", type=str, required=True, help="Path to the mask folder.")
    
    args = parser.parse_args()

    # Support multiple extensions
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files1 = sorted([f for f in os.listdir(args.folder1) if f.lower().endswith(valid_exts)])
    
    psnr_total = 0.0
    psnr_masked_total = 0.0
    count = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for f in tqdm(files1):
        p1 = os.path.join(args.folder1, f)
        p2 = os.path.join(args.folder2, f)
        
        # Mask path logic
        #p_mask = os.path.join(args.mask_folder, f)
        
        if not os.path.exists(p2):
            # Try finding p2 with different extension?
            # For now skip
            # print(f"Missing p2: {p2}") 
            continue

        # Handle mask extension mismatch
        # if not os.path.exists(p_mask):
        #     base_name = os.path.splitext(f)[0]
        #     found_mask = False
        #     for ext in valid_exts:
        #         test_path = os.path.join(args.mask_folder, base_name + ext)
        #         if os.path.exists(test_path):
        #             p_mask = test_path
        #             found_mask = True
        #             break
        #     if not found_mask:
        #         print(f"Warning: Mask not found for {f}, skipping.")
        #         continue

        try:
            img1 = read_image(p1).to(device)
            img2 = read_image(p2).to(device)
            #mask = read_mask(p_mask).to(device)
        except Exception as e:
            print(f"Error reading files for {f}: {e}")
            continue

        if img1.shape != img2.shape:
             img2 = torch.nn.functional.interpolate(img2, size=(img1.shape[2], img1.shape[3]), mode='bilinear', align_corners=False)
        
        # if mask.shape[2:] != img1.shape[2:]:
        #      mask = torch.nn.functional.interpolate(mask, size=(img1.shape[2], img1.shape[3]), mode='nearest')

        # Full PSNR
        val = psnr(img1, img2)
        psnr_total += val.item()

        # # Masked PSNR
        # val_masked = psnr_masked(img1, img2, mask)
        # psnr_masked_total += val_masked.item()
        
        count += 1

    if count > 0:
        avg_psnr = psnr_total / count
        #avg_psnr_masked = psnr_masked_total / count
        print(f"Average PSNR (Full) over {count} images: {avg_psnr:.4f}")
        #print(f"Average PSNR (Masked) over {count} images: {avg_psnr_masked:.4f}")
    else:
        print("No matching images/masks found.")

if __name__ == "__main__":
    main()
