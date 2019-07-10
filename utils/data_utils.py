import torch
import numpy as np
from skimage.measure import compare_ssim, compare_psnr

def torch2numpy(tensor, gamma=None):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    tensor *= 255.0
    return tensor.permute(0, 2, 3, 1).cpu().data.numpy()

def calculate_psnr(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    psnr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        psnr += compare_psnr(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...],
                                             data_range=255)
        n += 1.0
    return psnr / n


def calculate_ssim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    ssim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim += compare_ssim(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...],
                                             multichannel=True,
                                             data_range=255)
        n += 1.0
    return ssim / n