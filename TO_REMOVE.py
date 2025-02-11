#%%
import os
from diffusers import StableDiffusionPipeline
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch
import numpy as np
from tqdm import tqdm
from utils import get_data_superres, get_data_superres_BSRGAN
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTModel, ViTFeatureExtractor

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from StableDiff_LatentDiffusion import LatentDiffusion_superres

image_size = 192
magnification_factor = 4
batch_size = 1
Blur_radius = 0.5
vae_snapshot_name = "VAE_finetuning_MSE_Perceptual"
diffusion_snapshot_name = "Diffusion_finetuning"
dataset_path="celebA_100k"
device = 'cuda'
multiple_gpus = False

if image_size % magnification_factor != 0:
    raise ValueError('The image size must be a multiple of the magnification factor')

transform = transforms.Compose([
transforms.Resize((image_size, image_size)),
]) # The transforms.ToTensor() is in the get_data_superres function (in there
# first is applied this transform to y, then the resize according to the magnification_factor
# in order to get the x which is the lr_img and finally the to_tensor for both x
# and y is applied)

train_path = f'{dataset_path}/train_original'
valid_path = f'{dataset_path}/val_original'
test_path = f'{dataset_path}/test_original'

train_dataset = get_data_superres(train_path, magnification_factor, Blur_radius, False, 'PIL', transform)
val_dataset = get_data_superres(valid_path, magnification_factor, Blur_radius, False, 'PIL', transform)
test_dataset = get_data_superres(test_path, magnification_factor, Blur_radius, False, 'PIL', transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

VAE_weight_path = os.path.join('models_run', vae_snapshot_name+'.pt')
Diffusion_weight_path = os.path.join('models_run', diffusion_snapshot_name+'.pt')
latent_diff_model = LatentDiffusion_superres(VAE_weight_path=VAE_weight_path,
                                            Diffusion_weight_path=Diffusion_weight_path,
                                            device = device,
                                            multiple_gpus=multiple_gpus)

# %% EXAMPLE VAE ON celebA_100k
lr_img, hr_img = next(iter(test_loader))
lr_img = lr_img.to(device)
hr_img = hr_img.to(device)

latents = latent_diff_model.pipe.vae.encode(hr_img).latent_dist.sample()
reconstructed = latent_diff_model.pipe.vae.decode(latents).sample

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(lr_img[0].permute(1, 2, 0).cpu())
axs[0].set_title('Low Resolution Image')
axs[1].imshow(hr_img[0].permute(1, 2, 0).cpu())
axs[1].set_title('High Resolution Image')
axs[2].imshow(reconstructed[0].permute(1, 2, 0).cpu().detach().numpy())
axs[2].set_title('Reconstructed Image')
plt.show()

# %% EXAMPLE Stable Diffusion ON celebA_100k
# lr_img, hr_img = next(iter(test_loader))
# lr_img = lr_img.to(device)
# hr_img = hr_img.to(device)

sr_img = latent_diff_model.sample_superres(lr_img, 10)

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(lr_img[0].permute(1, 2, 0).cpu())
axs[0].set_title('Low Resolution Image')
axs[1].imshow(hr_img[0].permute(1, 2, 0).cpu())
axs[1].set_title('High Resolution Image')
axs[2].imshow(sr_img[0].permute(1, 2, 0).cpu().detach().numpy())
axs[2].set_title('Reconstructed Image')
plt.show()

# %% EXMPLE VAE ON up42
img_path = r'up42_sentinel2_patches\test_original\patch_0_4864.png'
img = Image.open(img_path).resize((image_size, image_size))
lr_img = img.resize((image_size//magnification_factor, image_size//magnification_factor))
transform = transforms.ToTensor()
hr_img = transform(img).unsqueeze(0).to(device)
lr_img = transform(lr_img).unsqueeze(0).to(device)

latents = latent_diff_model.pipe.vae.encode(hr_img).latent_dist.sample()
reconstructed = latent_diff_model.pipe.vae.decode(latents).sample

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(lr_img[0].permute(1, 2, 0).cpu())
axs[0].set_title('Low Resolution Image')
axs[1].imshow(hr_img[0].permute(1, 2, 0).cpu())
axs[1].set_title('High Resolution Image')
axs[2].imshow(reconstructed[0].permute(1, 2, 0).cpu().detach().numpy())
axs[2].set_title('Reconstructed Image')
plt.show()
# %%
