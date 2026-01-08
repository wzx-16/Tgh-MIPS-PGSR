import torch
import numpy as np
import argparse
import os
import sys
from PIL import Image

# Add the current directory to sys.path to allow imports
sys.path.append(os.getcwd())

try:
    from scene.gaussian_model import SphMipEncoding
except ImportError:
    print("Could not import SphMipEncoding. Make sure you are running this script from the project root.")
    sys.exit(1)

def visualize_sph_encoding(path, output_dir):
    print(f"Loading {path}...")
    try:
        # Load the entire object
        dir_encoding = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if isinstance(dir_encoding, SphMipEncoding):
        fm = dir_encoding.fm
    elif isinstance(dir_encoding, dict) and 'fm' in dir_encoding:
        print("Loaded object seems to be a state_dict.")
        fm = dir_encoding['fm']
    else:
        # Try to access fm attribute if it exists (duck typing)
        if hasattr(dir_encoding, 'fm'):
            fm = dir_encoding.fm
        else:
            print("Unknown format. Could not find 'fm' parameter.")
            return

    print(f"Feature map shape: {fm.shape}")
    # Expected shape: [Sn, dim, plane_size, 2*plane_size, feature_dim]
    
    # Handle dimensions
    if fm.dim() == 5:
        # Take the first element if Sn > 1 or dim > 1
        fm = fm[0, 0] 
    
    print(f"Processing shape: {fm.shape}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Visualize first 3 channels as RGB
    if fm.shape[-1] >= 3:
        rgb = fm[..., :3].detach().numpy()
        # Normalize to 0-255
        rgb_min = rgb.min()
        rgb_max = rgb.max()
        print(f"RGB range: {rgb_min} to {rgb_max}")
        
        if rgb_max > rgb_min:
            rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
        else:
            rgb_norm = rgb
            
        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
        
        Image.fromarray(rgb_uint8).save(os.path.join(output_dir, "sph_encoding_rgb.png"))
        print(f"Saved RGB visualization to {os.path.join(output_dir, 'sph_encoding_rgb.png')}")

    # Visualize each channel separately
    num_channels = fm.shape[-1]
    for i in range(num_channels):
        ch = fm[..., i].detach().numpy()
        ch_min = ch.min()
        ch_max = ch.max()
        
        if ch_max > ch_min:
            ch_norm = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = ch
            
        ch_uint8 = (ch_norm * 255).astype(np.uint8)
        
        Image.fromarray(ch_uint8).save(os.path.join(output_dir, f"sph_encoding_ch{i}.png"))
        # print(f"Saved channel {i} to {os.path.join(output_dir, f'sph_encoding_ch{i}.png')}")
    print(f"Saved {num_channels} individual channel images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SphMipEncoding from dir_encoding.pt")
    parser.add_argument("path", type=str, help="Path to dir_encoding.pt file")
    parser.add_argument("--output", type=str, default="sph_vis", help="Output directory")
    
    args = parser.parse_args()
    
    visualize_sph_encoding(args.path, args.output)
