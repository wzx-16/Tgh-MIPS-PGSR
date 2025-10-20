import os
import argparse
import glob

import numpy as np
import json
import sys
import math
import shutil
import sqlite3
import cv2
from PIL import Image
import numpy as np
from pathlib import Path

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def camTodatabase(txtfile, database_path):
    import os
    import argparse

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}

    if os.path.exists(database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

# ---- Utilities ----
def qvec2rotmat(qvec):
    # qvec = [qw, qx, qy, qz] (COLMAP order)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)

def K_from_colmap(model, params):
    # Returns a 3x3 K for common models. Distortion params are returned separately if you need them.
    p = list(map(float, params))
    if model == "SIMPLE_PINHOLE":           # [f, cx, cy]
        f, cx, cy = p
        return np.array([[f, 0, cx],
                         [0, f, cy],
                         [0, 0, 1]])
    elif model == "PINHOLE":                # [fx, fy, cx, cy]
        fx, fy, cx, cy = p
        return np.array([[fx, 0,  cx],
                         [0,  fy, cy],
                         [0,  0,  1]])
    elif model == "SIMPLE_RADIAL":          # [f, cx, cy, k1]
        f, cx, cy, *_ = p
        return np.array([[f, 0, cx],
                         [0, f, cy],
                         [0, 0, 1]])
    elif model == "RADIAL":                 # [f, cx, cy, k1, k2]
        f, cx, cy, *_ = p
        return np.array([[f, 0, cx],
                         [0, f, cy],
                         [0, 0, 1]])
    elif model == "OPENCV":                 # [fx, fy, cx, cy, k1, k2, p1, p2]
        fx, fy, cx, cy, *_ = p
        return np.array([[fx, 0,  cx],
                         [0,  fy, cy],
                         [0,  0,  1]])
    elif model == "OPENCV_FISHEYE":         # [fx, fy, cx, cy, k1, k2, k3, k4]
        fx, fy, cx, cy, *_ = p
        return np.array([[fx, 0,  cx],
                         [0,  fy, cy],
                         [0,  0,  1]])
    elif model == "FULL_OPENCV":            # [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
        fx, fy, cx, cy, *_ = p
        return np.array([[fx, 0,  cx],
                         [0,  fy, cy],
                         [0,  0,  1]])
    else:
        raise NotImplementedError(f"Camera model {model} not handled. Add its mapping to K here.")

# ---- Parsers ----
def read_cameras_txt(path):
    """
    Returns dict: camera_id -> {
        'model', 'width', 'height', 'params' (list of floats), 'K' (3x3 numpy)
    }
    """
    cams = {}
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln[0] == "#":
                continue
            # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
            parts = ln.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2]); height = int(parts[3])
            params = list(map(float, parts[4:]))
            K = K_from_colmap(model, params)
            cams[cam_id] = dict(model=model, width=width, height=height, params=params, K=K)
    return cams

def read_images_txt(path):
    """
    Returns list of dicts with:
      image_id, qvec(4), tvec(3), camera_id, name, R(3x3), t(3,), T_w2c(4x4), T_c2w(4x4), C(3,)
    """
    out = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and ln[0] != "#"]
    # images.txt is organized as pairs of lines: pose line, then 2D keypoint line (we skip keypoints)
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])))  # qw qx qy qz
        tvec = np.array(list(map(float, parts[5:8])))
        camera_id = int(parts[8])
        name = parts[9]
        R = qvec2rotmat(qvec)
        t = tvec.copy()                    # world -> cam
        # Transforms
        T_w2c = np.eye(4); T_w2c[:3,:3] = R; T_w2c[:3,3] = t
        T_c2w = np.eye(4); T_c2w[:3,:3] = R.T; T_c2w[:3,3] = -R.T @ t
        C = -R.T @ t                      # camera center in world
        out.append(dict(
            image_id=image_id, camera_id=camera_id, name=name,
            qvec=qvec, tvec=tvec,
            R=R, t=t, T_w2c=T_w2c, T_c2w=T_c2w, C=C
        ))
    return out

# ---- Example usage ----
def load_colmap_txt_model(model_dir):
    model_dir = Path(model_dir)
    cams = read_cameras_txt(model_dir / "cameras.txt")
    ims  = read_images_txt(model_dir / "images.txt")
    # Attach K and camera model info to each image entry
    for im in ims:
        cam = cams[im["camera_id"]]
        im["K"] = cam["K"]
        im["camera_model"] = cam["model"]
        im["size"] = (cam["width"], cam["height"])
        im["intrinsics_params"] = cam["params"]  # includes distortion if present
    return cams, ims

# Example:
# cams, images = load_colmap_txt_model("/path/to/sparse/0")
# print(images[0]["name"])
# print("K=\n", images[0]["K"])
# print("R=\n", images[0]["R"])
# print("t=", images[0]["t"])
# print("Camera center C=", images[0]["C"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser() # TODO: refine it.
    parser.add_argument("path", default="", help="input path to the video")
    args = parser.parse_args()

    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
        
    # extract images
    # videos = [os.path.join(args.path, 'videos', vname) for vname in os.listdir(os.path.join(args.path, 'videos')) if vname.endswith(".mp4")]
    images_path = os.path.join(args.path, "images/")
    images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    #images = [im for im in images if int(im[10:15]) < 2]

    with open(os.path.join(args.path, 'calibration_full.json'), 'r') as f:
        calib = json.load(f)
        
    poses = []
    Ks = []
    Dis = []
    mapxy = []
    cams = []
    cameras = calib
    camera_poses = calib
    N = len(cameras.keys())
    #S = np.diag([-1, -1, 1])
    for k in cameras.keys():
        # if int(k) % 5 != 0:
        #     continue
        cams.append(k)
        RT = np.eye(4)
        RT[:3, :3] = np.array(camera_poses[k]['R']).reshape(3,3)
        RT[:3, 3] = np.array(camera_poses[k]['T']).reshape(3)
        RT = np.linalg.inv(RT)  # convert to world to camera
        W, H = cameras[k]['imgSize'][0], cameras[k]['imgSize'][1]
        poses.append(RT)
        K = np.array(cameras[k]['K']).reshape(3,3)
        Dis.append(np.array(cameras[k]['distCoeff']))
        #new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, Dis[-1], (W, H), 1, (W, H))
        print(k)
        #print(new_camera_matrix)
        # mapx, mapy = cv2.initUndistortRectifyMap(K, Dis[-1], None, new_camera_matrix, (W, H), cv2.CV_32FC1)
        # new_camera_matrix[:2] /= 8.0  # scale down by 4x
        # new_camera_matrix[0, 2] -= 120
        # new_camera_matrix[1, 2] -= 180
        # new_camera_matrix[0, 2] -= 50
        # new_camera_matrix[1, 2] -= 100
        # W, H = W - 100, H - 200

        # new_camera_matrix[0][0] /= 2
        # new_camera_matrix[1][1] /= 2
        # new_camera_matrix[0][2] /= 2
        # new_camera_matrix[1][2] /= 2
        # W = W // 2
        # H = H // 2
        # W, H = W - 240, H - 360
        Ks.append(K)
        # W, H = W//8, H//8
        # for imagename in images:
        #     filename = imagename.split("/")[1]
        #     if filename.endswith(".jpg"):
        #         full_path = os.path.join(images_path, filename)
        #         img = Image.open(full_path)
        #         left = 120
        #         top = 180
        #         right = left + W
        #         bottom = top + H
        #         cropped_img = img.crop((left, top, right, bottom))
        #         resized_img = cropped_img.resize((cropped_img.width // 2, cropped_img.height // 2))
        #         resized_img.save(full_path)
        #         print(filename)
    print(cams)
    poses = np.stack(poses, axis = 0)
    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    #poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)
    # to homogeneous 
    # last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    # poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

    # the following stuff are from colmap2nerf... 
    #poses = np.stack(poses, axis = 0)
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    #poses = poses[:, [1, 0, 2, 3], :] # swap y and z
    #poses[:, 2, :] *= -1 # flip whole world upside down

    # up = poses[:, 0:3, 1].sum(0)
    # up = up / np.linalg.norm(up)
    # R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    # R = np.pad(R, [0, 1])
    # R[-1, -1] = 1

    # poses = R @ poses

    # totw = 0.0
    # totp = np.array([0.0, 0.0, 0.0])
    # for i in range(N):
    #     mf = poses[i, :3, :]
    #     for j in range(i + 1, N):
    #         mg = poses[j, :3, :]
    #         p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
    #         #print(i, j, p, w)
    #         if w > 0.01:
    #             totp += p * w
    #             totw += w
    # totp /= totw
    # print(f'[INFO] totp = {totp}')
    # poses[:, :3, 3] -= totp

    # avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()

    # poses[:, :3, 3] *= 4.0 / avglen

    #print(f'[INFO] average radius = {avglen}')
    
    train_frames = []
    test_frames = []
    #time_offset = [0.0, 0.0, 0.0, 0.015, 0.015, 0.0, 0.015, -0.015, 0.0, 0.015, 0.015, 0.005, 0.005]
    _, ims = load_colmap_txt_model(args.path)
    # print(ims)
    for im in ims:
        im["T_c2w"][0:3, 1] *= -1
        im["T_c2w"][0:3, 2] *= -1
    for i in range(N):
        cam_frames = [{'file_path': "images/" + im["name"][0:8], 
                       'fl_x': im["K"][0, 0],
                       'fl_y': im["K"][1, 1],
                       'cx': im["K"][0, 2],
                       'cy': im["K"][1, 2],
                       'transform_matrix': im["T_c2w"].tolist(),
                       'time': (int(im["name"].split('.')[0][-4:]) / 30.)} for im in ims if cams[i] == im["name"][0:2] and int(im["name"][0:2]) != 11]
        if i == 7:
        #if i == -1:
            test_frames += cam_frames
        else:
            train_frames += cam_frames

    train_transforms = {
        'w': W,
        'h': H,
        'fl_x': ims[0]["K"][0,0],
        'fl_y': ims[0]["K"][1,1],
        'cx': ims[0]["K"][0,2],
        'cy': ims[0]["K"][1,2],
        'frames': train_frames,
    }
    test_transforms = {
        'w': W,
        'h': H,
        'fl_x': ims[0]["K"][0,0],
        'fl_y': ims[0]["K"][1,1],
        'cx': ims[0]["K"][0,2],
        'cy': ims[0]["K"][1,2],
        'frames': test_frames,
    }

    train_output_path = os.path.join(args.path, 'transforms_train.json')
    test_output_path = os.path.join(args.path, 'transforms_test.json')
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)