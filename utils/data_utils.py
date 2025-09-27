import os
import cv2
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            with Image.open(viewpoint_cam.image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))#[100:-100, 100:-100]
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
            image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image_load = np.array(image_load)
            #image_load = cv2.resize(image_load, (image_load.shape[1]//2, image_load.shape[0]//2), interpolation=cv2.INTER_LINEAR)
            image_load = torch.from_numpy(image_load) / 255.0
            resized_image_rgb = image_load.permute(2, 0, 1)
            # resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
            viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                viewpoint_image *= gt_alpha_mask
            #else:
                #viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
                
            # mask_path = "/" + os.path.join(os.path.join(*viewpoint_cam.image_path.split("/")[0:-2]), os.path.join("mattings", viewpoint_cam.image_name.split("_")[0]))
            # mask_name = viewpoint_cam.image_name.split("_")[-1].split(".")[0] + ".png"
            # with Image.open(os.path.join(mask_path, mask_name)) as image_load:
            #     #loaded_mask_PIL = image_load.resize((1500, 2000))
            #     loaded_mask_PIL = image_load.convert("L")
            # loaded_mask = torch.from_numpy(np.array(loaded_mask_PIL)).unsqueeze(0) / 255.0
            loaded_mask = None

            # pals = os.path.split(viewpoint_cam.image_path)
            # pa = '/media/bbnc/Elements/test/normal/' + pals[-1][3:7] + '/' + pals[-1].replace('jpg', 'png')
            # # pa_mask = '/media/bbnc/Elements/test/human_masks/' + pals[-1][3:7] + '/' + pals[-1].replace('jpg', 'png')
            # # mask = torch.from_numpy(cv2.imread(pa_mask)) > 0
            # # if len(mask.shape) > 2:
            # #     mask = mask[..., 0]
            # # if 1200 > int(pals[-1][-10:-4]) > 901:
            # #     rr = 255 - cv2.imread(pa)[100:-100, 100:-100, [2, 1, 0]]
            # #     rr = cv2.resize(rr, (rr.shape[1]//2, rr.shape[0]//2), interpolation=cv2.INTER_NEAREST)
            # #     rr = torch.from_numpy(rr).float().permute(2, 0, 1) / 255.0
            # #     rr = rr * 2 -1
            # # else:
            rr = None
                
        else:
            viewpoint_image = viewpoint_cam.image
            
        return viewpoint_image, loaded_mask, viewpoint_cam, rr
    
    def __len__(self):
        return len(self.viewpoint_stack)
    
