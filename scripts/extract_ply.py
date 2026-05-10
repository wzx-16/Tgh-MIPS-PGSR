from pathlib import Path
import shutil

# Folder that contains the numbered folders, e.g. 1100/sparse.ply
source_root = Path("/root/autodl-tmp/projects/4dRefgs_sc/data/selfcap_floor2/sparse_points_per_timestamp")

# New folder where renamed .ply files will go
output_dir = Path("/root/autodl-tmp/projects/4dRefgs_sc/data/selfcap_floor2/pcds_j10")
output_dir.mkdir(exist_ok=True)

for ply_path in source_root.glob("*/sparse.ply"):
    folder_name = ply_path.parent.name

    # Only process folders with numeric names
    if not folder_name.isdigit():
        continue

    # Rename 1100 -> 01100.ply, 7 -> 00007.ply
    new_name = f"{int(folder_name):05d}.ply"

    destination = output_dir / new_name
    shutil.copy2(ply_path, destination)

    print(f"Copied {ply_path} -> {destination}")