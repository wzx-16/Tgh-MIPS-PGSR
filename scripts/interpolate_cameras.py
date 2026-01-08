import sys
import os
import numpy as np
import collections
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

# Define Image namedtuple structure as used in COLMAP
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def read_extrinsics_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                
                # Check if the next line actually contains points. 
                # Sometimes images.txt might not have points if it was manually created or modified,
                # but valid COLMAP text format requires alternating lines.
                # If the split is empty, it means no points.
                
                if len(elems) > 0:
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                           tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                else:
                    xys = None
                    point3D_ids = None

                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def create_interpolated_cameras(input_path, output_path, multiplier=4):
    print(f"Reading from {input_path}")
    images = read_extrinsics_text(input_path)
    
    if not images:
        print("No images found or error reading file.")
        return

    # Sort images by name
    sorted_images = sorted(images.values(), key=lambda x: x.name)
    print(f"Found {len(sorted_images)} images. Sorting by name.")
    
    qvecs = np.array([img.qvec for img in sorted_images]) # (N, 4) -> (w, x, y, z)
    tvecs = np.array([img.tvec for img in sorted_images]) # (N, 3)
    camera_ids = [img.camera_id for img in sorted_images]
    
    num_frames = len(sorted_images)
    new_num_frames = (num_frames - 1) * multiplier + 1
    print(f"Interpolating from {num_frames} frames to {new_num_frames} frames (multiplier {multiplier}).")
    
    t = np.arange(num_frames)
    new_t = np.linspace(0, num_frames - 1, new_num_frames)
    
    # Position interpolation
    # Use cubic spline if we have enough points, else linear
    kind = 'cubic' if num_frames >= 4 else 'linear'
    print(f"Using {kind} interpolation for position.")
    interp_func_tvec = interp1d(t, tvecs, axis=0, kind=kind)
    new_tvecs = interp_func_tvec(new_t)
    
    # Rotation interpolation (Slerp)
    # COLMAP quaternions are (w, x, y, z)
    # SciPy expects (x, y, z, w)
    rot_quats = qvecs[:, [1, 2, 3, 0]] 
    rotations = R.from_quat(rot_quats)
    slerp = Slerp(t, rotations)
    new_rotations = slerp(new_t)
    new_quats = new_rotations.as_quat() # (x, y, z, w)
    
    # Convert back to (w, x, y, z)
    new_qvecs = new_quats[:, [3, 0, 1, 2]]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Writing output
    print(f"Writing to {output_path}")
    with open(output_path, "w") as f:
        f.write("# Image list with two lines of data per image.\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {new_num_frames}, mean observations per image: 0\n")
        
        for i in range(new_num_frames):
            image_id = i + 1
            qw, qx, qy, qz = new_qvecs[i]
            tx, ty, tz = new_tvecs[i]
            camera_id = camera_ids[0] # Using first camera ID
            name = f"interpolated_{i:05d}.jpg"
            
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n")
            f.write("\n") # Empty points

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/interpolate_cameras.py <input_images.txt> <output_images.txt> [multiplier]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    multiplier = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    create_interpolated_cameras(input_path, output_path, multiplier)
