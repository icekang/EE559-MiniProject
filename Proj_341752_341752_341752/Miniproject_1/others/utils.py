import torch

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio: denoised and ground_truth have rage [0, 1]
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 1e-08)