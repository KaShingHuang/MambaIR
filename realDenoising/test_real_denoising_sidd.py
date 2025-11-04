import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn

from basicsr.models.archs.mambairunet_arch import MambaIRUNet
import scipy.io as sio
from PIL import Image

parser = argparse.ArgumentParser(description='Real Image Denoising')

parser.add_argument('--input_dir', default='/data2/CarnegieBin_data/HuangJiaCheng/AST-main/LSUI/patch/test/input', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Real_Denoising/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/data2/CarnegieBin_data/HuangJiaCheng/MambaIR/realDenoising/experiments/MambaIR_RealDN/models/net_g_24000.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
opt_str = r"""
  type: MambaIRUNet
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  mlp_ratio: 1.5
  bias: False
  dual_pixel_task: False
"""

import yaml
x = yaml.safe_load(opt_str)

s = x.pop('type')
##########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
  result_dir_png = os.path.join(args.result_dir, 'png','patch')
  os.makedirs(result_dir_png, exist_ok=True)

model_restoration = MambaIRUNet(**x)

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Process data
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
img_files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                    if f.lower().endswith(img_exts)])

if len(img_files) == 0:
    # 如果没有找到图像文件，尝试原先的 .mat 加载方式（保持兼容）
    filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
    img = sio.loadmat(filepath)
    Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
    Inoisy /= 255.
    restored = np.zeros_like(Inoisy)
    with torch.no_grad():
        for i in tqdm(range(Inoisy.shape[0]), desc='mat images'):
            for k in range(Inoisy.shape[1]):
                noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                restored_patch = model_restoration(noisy_patch)
                restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
                restored[i, k, :, :, :] = restored_patch
                if args.save_images:
                    out_uint8 = (restored_patch * 255.0).round().astype(np.uint8)
                    Image.fromarray(out_uint8).save(os.path.join(result_dir_png, f'{i+1:04d}_{k+1:02d}.png'))
    sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})
else:
    restored_list = []
    with torch.no_grad():
        for idx, fp in enumerate(tqdm(img_files, desc='Processing images')):
            img_pil = Image.open(fp).convert('RGB')
            arr = np.asarray(img_pil).astype(np.float32) / 255.0
            inp = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
            out = model_restoration(inp)
            out = torch.clamp(out, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
            restored_list.append(out)

            if args.save_images:
                out_uint8 = (out * 255.0).round().astype(np.uint8)
                Image.fromarray(out_uint8).save(os.path.join(result_dir_png, os.path.basename(fp)))

    # 保存为 mat：如果所有图片尺寸一致则保存为一个大数组，否则分别保存为单独 mat
    try:
        shapes = [im.shape for im in restored_list]
        if len(set(shapes)) == 1:
            restored_arr = np.stack(restored_list, axis=0)  # (N, H, W, C)
            sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored_arr})
        else:
            for i, im in enumerate(restored_list):
                sio.savemat(os.path.join(result_dir_mat, f'Idenoised_{i+1:04d}.mat'), {"Idenoised": im})
     except Exception as e:
        print('Warning saving mat:', e)
