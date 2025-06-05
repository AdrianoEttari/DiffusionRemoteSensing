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

# %% EXAMPLE VAE ON up42
img_path = os.path.join('up42_sentinel2_patches','test_original','patch_0_4864.png')
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
# %% FINE-TUNE VAE EXAMPLE ON up42
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from utils import get_data_superres, get_data_superres_BSRGAN, video_maker, CosineAnnealingWarmupRestarts
from UNet_model_superres_VMHA import Residual_Attention_UNet_superres, Residual_VisionMultiheadAttention_UNet_superres, Residual_DiffiT_UNet_superres, EMA
from ViT_model import ViTModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

model_path = "CompVis/stable-diffusion-v1-4"
snapshot_path = os.path.join('models_run','VAE_up42_hr256_finetuning')
device = 'mps'
pipe = StableDiffusionPipeline.from_pretrained(model_path)
vae_model = pipe.vae
vae_model = vae_model.eval()
vae_model = vae_model.to(device)
transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

def psnr(ground_truth, predicted, pixel_max=255):
    '''
    Compute the Peak Signal to Noise Ratio between the real mask and the predicted one.

    The masks must be float32 and not uint8, because the second is 8 bit and so 
    has just values between 0 and 255.
    '''
    ground_truth = ground_truth.astype(np.float32)  # Convert to float
    predicted = predicted.astype(np.float32)
    mse = np.mean((ground_truth - predicted) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match should return infinity
    return 10 * np.log10(pixel_max**2 / mse)

def _load_snapshot_VAE(snapshot_path, model):
    '''
    This function loads the model state and the last epoch of training (so that we can restart the
    training at this point instead of restarting from 0) from a snapshot.
    It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
    it from the last snapshot.
    '''
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    model.load_state_dict(snapshot)

    print(f"Snapshot loaded from {snapshot_path}")

_load_snapshot_VAE(snapshot_path, vae_model)

img = Image.open(os.path.join('up42_sentinel2_patches','test_original','patch_256_1792.png'))
img = transform(img).unsqueeze(0).to(device)
encoded_img = vae_model.encode(img).latent_dist.sample()
decoded_img = vae_model.decode(encoded_img).sample

fig, axs = plt.subplots(1,3, figsize=(10,5))
axs = axs.ravel()
axs[0].imshow(img[0].permute(1,2,0).cpu())
axs[0].set_title('Original Image')
axs[1].imshow(encoded_img[0].permute(1,2,0).detach().cpu())
axs[1].set_title('Encoded Image')
axs[2].imshow(decoded_img[0].permute(1,2,0).detach().cpu())
axs[2].set_title('Decoded Image')   
plt.show()
print(psnr(img[0].permute(1,2,0).cpu().numpy(), decoded_img[0].permute(1,2,0).detach().cpu().numpy(), pixel_max=1))
# # %%
# encoded_img1 = vae_model.encode(img).latent_dist.sample()
# encoded_img2 = vae_model.encode(img).latent_dist.sample()
# encoded_img3 = vae_model.encode(img).latent_dist.sample()
# # %%
# encoded_img4 = vae_model.encoder(img)
# decoded_img4 = vae_model(img).sample
#%% VAE SAR to NDVI
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import get_data_SAR_TO_NDVI, CosineAnnealingWarmupRestarts
from UNet_model_superres_CrossAttention import Residual_CrossAttention_UNet_superres
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from torch import nn

def VAE_model_maker(device, freeze_vae_params):
    vae_model_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(vae_model_path)
    vae_model = pipe.vae.to(device)
    vae_model = VAE_model_wrapped(vae_model, in_channels=1, freeze_vae_params=freeze_vae_params).to(device)
    return vae_model
class VAE_model_wrapped(nn.Module):
    def __init__(self, vae_model, in_channels, freeze_vae_params=False):
        super(VAE_model_wrapped, self).__init__()
        self.vae_model = vae_model
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1)
        if freeze_vae_params: #if True, the vae parameters are frozen and just the conv layer is trained
            for param in self.vae_model.parameters():
                param.requires_grad = False
    def encode(self, x):
        x = self.conv(x)
        latents = self.vae_model.encode(x)
        return latents

    def decode(self, latents):
        return self.vae_model.decode(latents)

def _load_snapshot_VAE(snapshot_path, model):
    '''
    This function loads the model state and the last epoch of training (so that we can restart the
    training at this point instead of restarting from 0) from a snapshot.
    It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
    it from the last snapshot.
    '''
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    model.load_state_dict(snapshot)

    print(f"Snapshot loaded from {snapshot_path}")

device='cuda'
snapshot_path = os.path.join('models_run','VAE_SAR_TO_NDVI_finetuning_gradientAccumulation.pt')
vae_model = VAE_model_maker(device=device, freeze_vae_params=False)
vae_model.eval()
vae_model = vae_model.to(device)
_load_snapshot_VAE(snapshot_path, vae_model)

data_path = os.path.join("SAR_TO_NDVI_dataset", "test")
test_data = get_data_SAR_TO_NDVI(data_path)

sar_img, ndvi_img = test_data[0]
sar_img = sar_img.unsqueeze(0).to(device)
ndvi_img = ndvi_img.unsqueeze(0).to(device)

encoded_sar_img = vae_model.encode(sar_img).latent_dist.sample()
decoded_sar_img = vae_model.decode(encoded_sar_img).sample

encoded_ndvi_img = vae_model.encode(ndvi_img).latent_dist.sample()
decoded_ndvi_img = vae_model.decode(encoded_ndvi_img).sample

fig, axs = plt.subplots(2,3, figsize=(10,5))
axs = axs.ravel()

axs[0].imshow(encoded_sar_img[0,:3,:,:].permute(1,2,0).detach().cpu())
axs[0].set_title("encoded_sar_img")
axs[1].imshow(decoded_sar_img[0,:,:,:].permute(1,2,0).detach().cpu())
axs[1].set_title("decoded_sar_img")
axs[2].imshow(sar_img[0,:,:,:].permute(1,2,0).detach().cpu())
axs[2].set_title("sar_img")
axs[3].imshow(encoded_ndvi_img[0,:3,:,:].permute(1,2,0).detach().cpu())
axs[3].set_title("encoded_ndvi_img")
axs[4].imshow(decoded_ndvi_img[0,:,:,:].permute(1,2,0).detach().cpu())
axs[4].set_title("decoded_ndvi_img")
axs[5].imshow(ndvi_img[0,:,:,:].permute(1,2,0).detach().cpu())
axs[5].set_title("ndvi_img")

#%% VAE 102flowers generation
from generate_new_imgs.utils import dataset_maker
from generate_new_imgs.UNet_model_generation_CrossAttention import Residual_Attention_UNet_generation
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from torch import nn

def VAE_model_maker(device):
    vae_model_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(vae_model_path)
    vae_model = pipe.vae.to(device)
    return vae_model

def _load_snapshot_VAE(snapshot_path, model):
    '''
    This function loads the model state and the last epoch of training (so that we can restart the
    training at this point instead of restarting from 0) from a snapshot.
    It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
    it from the last snapshot.
    '''
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    model.load_state_dict(snapshot)

    print(f"Snapshot loaded from {snapshot_path}")

device='cuda'
snapshot_path = os.path.join('models_run','VAE_102flowers_finetuning_gradientAccumulation.pt')
vae_model = VAE_model_maker(device=device)
vae_model.eval()
vae_model = vae_model.to(device)
_load_snapshot_VAE(snapshot_path, vae_model)

image_size = 512
dataset_path = os.path.join("102flowers_dataset")

dataset = dataset_maker(image_size, dataset_path)

flower_0_img = dataset[100][0].unsqueeze(0).to(device)
with torch.no_grad():
    encoded_img = vae_model.encode(flower_0_img).latent_dist.sample()
    decoded_img = vae_model.decode(encoded_img).sample

fig, axs = plt.subplots(1,3)
axs = axs.ravel()

axs[0].imshow(flower_0_img[0].permute(1,2,0).detach().cpu())
axs[1].imshow(encoded_img[0,:3,:,:].permute(1,2,0).detach().cpu())
axs[2].imshow(decoded_img[0].permute(1,2,0).detach().cpu())
plt.show()
# %% LEARNING RATE SCHEDULE EXAMPLE
from UNet_model_superres_CrossAttention import Residual_CrossAttention_UNet_superres
import torch
from utils import get_data_superres, get_data_superres_BSRGAN, video_maker, CosineAnnealingWarmupRestarts
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

image_size = 256
device='cuda'
dataset_path = "up42_sentinel2_patches"
lr = 1e-4
magnification_factor = 4
batch_size = 1
Blur_radius = 0.5
multiple_gpus = False

model = Residual_CrossAttention_UNet_superres(3, 3, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=15,
                cycle_mult=2,
                max_lr=lr,
                min_lr=1e-5,
                warmup_steps=5,
                gamma=0.9
            )
# from torch.optim.lr_scheduler import CosineAnnealingLR
# scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
iters = len(train_loader)
learning_rate = []
# for epoch in range(5):
#     for i, sample in enumerate(train_loader):
#         scheduler.step()
#         learning_rate.append(optimizer.param_groups[0]['lr'])
for epoch in range(100):
    scheduler.step()
    learning_rate.append(optimizer.param_groups[0]['lr'])

plt.plot(learning_rate)
plt.show()
# %% FINE-TUNE VAE HRandLR EXAMPLE ON up42
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from utils import get_data_superres, get_data_superres_BSRGAN, video_maker, CosineAnnealingWarmupRestarts
from UNet_model_superres_CrossAttention import Residual_CrossAttention_UNet_superres
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

model_path = "CompVis/stable-diffusion-v1-4"
snapshot_path = os.path.join('models_run','VAE_up42_LRandHR_finetuning_gradientAccumulation.pt')
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_path)
vae_model = pipe.vae
vae_model = vae_model.eval()
vae_model = vae_model.to(device)
transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

def psnr(ground_truth, predicted, pixel_max=255):
    '''
    Compute the Peak Signal to Noise Ratio between the real mask and the predicted one.

    PSNR = 10 * log10((pixel_max^2) / MSE) is a metric used to measure the quality
    of reconstruction of an image compared to its original version.
    
    The masks must be float32 and not uint8, because the second is 8 bit and so 
    has just values between 0 and 255.
    '''
    ground_truth = ground_truth.astype(np.float32)  # Convert to float
    predicted = predicted.astype(np.float32)
    mse = np.mean((ground_truth - predicted) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match should return infinity
    return 10 * np.log10(pixel_max**2 / mse)

def _load_snapshot_VAE(snapshot_path, model):
    '''
    This function loads the model state and the last epoch of training (so that we can restart the
    training at this point instead of restarting from 0) from a snapshot.
    It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
    it from the last snapshot.
    '''
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    model.load_state_dict(snapshot)

    print(f"Snapshot loaded from {snapshot_path}")

_load_snapshot_VAE(snapshot_path, vae_model)

img = Image.open(os.path.join('up42_sentinel2_patches','test_original','patch_256_1792.png'))
img = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    encoded_img = vae_model.encode(img).latent_dist.sample()
    decoded_img = vae_model.decode(encoded_img).sample

fig, axs = plt.subplots(1,3, figsize=(10,5))
axs = axs.ravel()
axs[0].imshow(img[0].permute(1,2,0).cpu())
axs[0].set_title('Original Image')
axs[1].imshow(encoded_img[0][:3,:,:].permute(1,2,0).detach().cpu())
axs[1].set_title('Encoded Image')
axs[2].imshow(decoded_img[0].permute(1,2,0).detach().cpu())
axs[2].set_title('Decoded Image')   
plt.show()
print(psnr(img[0].permute(1,2,0).cpu().numpy(), decoded_img[0].permute(1,2,0).detach().cpu().numpy(), pixel_max=1))

# %% FINE-TUNE VAE HRandLR EXAMPLE ON celebA_100k
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from utils import get_data_superres, get_data_superres_BSRGAN, video_maker, CosineAnnealingWarmupRestarts
from UNet_model_superres_VMHA import Residual_Attention_UNet_superres, Residual_VisionMultiheadAttention_UNet_superres, Residual_DiffiT_UNet_superres, EMA
from ViT_model import ViTModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

model_path = "CompVis/stable-diffusion-v1-4"
snapshot_path = os.path.join('models_run','VAE_celeb100k_LRandHR_finetuning_gradientAccumulation')
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_path)
vae_model = pipe.vae
vae_model = vae_model.eval()
vae_model = vae_model.to(device)
transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

def psnr(ground_truth, predicted, pixel_max=255):
    '''
    Compute the Peak Signal to Noise Ratio between the real mask and the predicted one.

    The masks must be float32 and not uint8, because the second is 8 bit and so 
    has just values between 0 and 255.
    '''
    ground_truth = ground_truth.astype(np.float32)  # Convert to float
    predicted = predicted.astype(np.float32)
    mse = np.mean((ground_truth - predicted) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match should return infinity
    return 10 * np.log10(pixel_max**2 / mse)

def _load_snapshot_VAE(snapshot_path, model):
    '''
    This function loads the model state and the last epoch of training (so that we can restart the
    training at this point instead of restarting from 0) from a snapshot.
    It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
    it from the last snapshot.
    '''
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    model.load_state_dict(snapshot)

    print(f"Snapshot loaded from {snapshot_path}")

_load_snapshot_VAE(snapshot_path, vae_model)

img = Image.open(os.path.join("celebA_100k","test_original","052120.jpg"))
img = transform(img).unsqueeze(0).to(device)
encoded_img = vae_model.encode(img).latent_dist.sample()
decoded_img = vae_model.decode(encoded_img).sample


fig, axs = plt.subplots(1,3, figsize=(10,5))
axs = axs.ravel()
axs[0].imshow(img[0].permute(1,2,0).cpu())
axs[0].set_title('Original Image')
axs[1].imshow(encoded_img[0][:3,:,:].permute(1,2,0).detach().cpu())
axs[1].set_title('Encoded Image')
axs[2].imshow(decoded_img[0].permute(1,2,0).detach().cpu())
axs[2].set_title('Decoded Image')   
plt.show()
print(psnr(img[0].permute(1,2,0).cpu().numpy(), decoded_img[0].permute(1,2,0).detach().cpu().numpy(), pixel_max=1))

# %% DIFFUSION MODEL ON celebA_100k
from train_diffusion_superres import Diffusion
from UNet_model_superres_VMHA import Residual_Attention_UNet_superres
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from UNet_model_superres_VMHA import Residual_Attention_UNet_superres
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

noise_schedule='cosine'
noise_steps=1000
model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart"
ema_smoothing=False
Degradation_type='downblur'
input_channels = output_channels = 4
device = 'cuda'
VAE_weight_path = os.path.join('models_run','VAE_celeb100k_LRandHR_finetuning_gradientAccumulation')
image_size=256
multiple_gpus=False
magnification_factor=4
model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
model_path = "CompVis/stable-diffusion-v1-4"
snapshot_folder_path = os.path.join(os.curdir, 'models_run', model_name, 'weights')
snapshot_path = os.path.join(snapshot_folder_path, "snapshot.pt")
pipe = StableDiffusionPipeline.from_pretrained(model_path)
vae_model = pipe.vae
vae_model = vae_model.eval()
vae_model = vae_model.to(device)

diffusion = Diffusion(
    noise_schedule=noise_schedule, model=model, vae_model=vae_model,
    snapshot_path=snapshot_path,
    VAE_weight_path=VAE_weight_path,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name, Degradation_type=Degradation_type,
    multiple_gpus=multiple_gpus, ema_smoothing=ema_smoothing)
    
lr_img = Image.open(os.path.join("celebA_100k","test_original","052120.jpg"))
transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])
lr_img = transform(lr_img).to(device)

latent_lr_img, latent_sr_img, superres_img = diffusion.sample(n=1,model=model, lr_img=lr_img, input_channels=img.shape[0], generate_video=False)

fig, axs = plt.subplots(1,5, figsize=(15,15))
axs = axs.ravel()
axs[0].imshow(lr_img.permute(1,2,0).cpu())
axs[0].set_title('Original Image')
axs[1].imshow(encoded_img[0][:3,:,:].permute(1,2,0).detach().cpu())
axs[1].set_title('Encoded Image')
axs[2].imshow(decoded_img[0].permute(1,2,0).detach().cpu())
axs[2].set_title('Decoded Image')   
axs[3].imshow(superres_img[0].permute(1,2,0).detach().cpu())
axs[3].set_title('Super Resolution Image')
axs[4].imshow(latent_sr_img[0][:3,:,:].permute(1,2,0).detach().cpu())
axs[4].set_title('Super Resolution Latent')
plt.show()

# %% SENTINEL 2 BIG IMAGE PROCESSING
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

blue_channel_60m = rasterio.open(os.path.join("Napoli_sentinel2","R60m","T33TVF_20250205T095231_B02_60m.jp2")).read(1).astype(np.float32)[:,:, None]
green_channel_60m = rasterio.open(os.path.join("Napoli_sentinel2","R60m","T33TVF_20250205T095231_B03_60m.jp2")).read(1).astype(np.float32)[:,:, None]
red_channel_60m = rasterio.open(os.path.join("Napoli_sentinel2","R60m","T33TVF_20250205T095231_B04_60m.jp2")).read(1).astype(np.float32)[:,:, None]
scl_channel_60m = rasterio.open(os.path.join("Napoli_sentinel2","R60m","T33TVF_20250205T095231_SCL_60m.jp2")).read(1).astype(np.float32)[:,:, None]
# cloud_mask = (scl_channel_60m == 3) | (scl_channel_60m == 8) | (scl_channel_60m == 9) | (scl_channel_60m == 10) | (scl_channel_60m == 11)

SCALE_FACTOR = 10000.0  # for L2A (use 1E4 for L1C)


# Convert DN to reflectance
blue_channel_60m = np.clip(blue_channel_60m / SCALE_FACTOR, 0,1)
green_channel_60m = np.clip(green_channel_60m / SCALE_FACTOR, 0,1)
red_channel_60m = np.clip(red_channel_60m / SCALE_FACTOR, 0,1)

# blue_channel_60m[cloud_mask] = np.nan
# red_channel_60m[cloud_mask] = np.nan
# green_channel_60m[cloud_mask] = np.nan

rgb_60m = np.concatenate([red_channel_60m, green_channel_60m, blue_channel_60m], axis=2)

rgb_60m = rgb_60m[:1024, :1024,:]*255
rgb_60m = rgb_60m.astype(np.uint8)
Image.fromarray(rgb_60m).save('rgb_60m.png')

plt.imshow(rgb_60m)
plt.title("Sentinel-2 RGB")
plt.show()

# %% SAR to NDVI trials
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

len(os.listdir(os.path.join("SAR_TO_NDVI_dataset","train","opt")))

sar_img = torch.load(os.path.join("SAR_TO_NDVI_dataset","test","sar","Victoria_0_20180130_patch_69.pt"))
sar_img = sar_img[0].unsqueeze(0).permute(1,2,0).cpu().numpy()
plt.imshow(sar_img)
# %% SCALE the latents images for correctly train the diffusion model
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from utils import get_data_superres, get_data_superres_BSRGAN, video_maker, CosineAnnealingWarmupRestarts
from UNet_model_superres_CrossAttention import Residual_CrossAttention_UNet_superres
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from tqdm import tqdm 

model_path = "CompVis/stable-diffusion-v1-4"
snapshot_path = os.path.join('models_run','VAE_up42_LRandHR_finetuning_gradientAccumulation.pt')
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_path)
vae_model = pipe.vae
vae_model = vae_model.eval()
vae_model = vae_model.to(device)
transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.Resize((256, 256), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

def _load_snapshot_VAE(snapshot_path, model):
    '''
    This function loads the model state and the last epoch of training (so that we can restart the
    training at this point instead of restarting from 0) from a snapshot.
    It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
    it from the last snapshot.
    '''
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    model.load_state_dict(snapshot)

    print(f"Snapshot loaded from {snapshot_path}")

_load_snapshot_VAE(snapshot_path, vae_model)

test_path = os.path.join("up42_sentinel2_patches","test_original")

means = []
means_adj = []
vars = []
vars_adj = []

for img_name in tqdm(os.listdir(test_path)):
    img = Image.open(os.path.join(test_path,img_name))
    img = transform(img).unsqueeze(0).to(device)
    encoded_img = vae_model.encode(img).latent_dist.sample()
    encoded_img_adj = encoded_img.clone() * 0.18215
    means.append(encoded_img[0].mean().item())
    means_adj.append(encoded_img_adj[0].mean().item())
    vars.append(encoded_img[0].var().item())
    vars_adj.append(encoded_img_adj[0].var().item())
    
plt.hist(vars, bins=100, alpha=0.5, label='Original')
plt.hist(vars_adj, bins=100, alpha=0.5, label='Adjusted')
plt.xlabel('Variance')
plt.ylabel('Frequency')
plt.title('Histogram of Variance')
plt.legend()
plt.show()

plt.hist(means, bins=100, alpha=0.5, label='Original')
plt.hist(means_adj, bins=100, alpha=0.5, label='Adjusted')
plt.xlabel('Mean')
plt.ylabel('Frequency')
plt.title('Histogram of Mean')
plt.legend()
plt.show()

#%% LR vs HR vs SR(latent_MAE) vs SR(latent_CLIP) vs SR(bicubic)
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import get_data_superres, CosineAnnealingWarmupRestarts
from PIL import Image
import numpy as np
from train_diffusion_superres import *
from UNet_model_superres_CrossAttention import Residual_CrossAttention_UNet_superres


device = 'cuda'
noise_schedule="cosine"
UNet_type="residual cross attention unet"
input_channels=output_channels=4
image_size=256
magnification_factor=4
VAE_model_name="VAE_up42_LRandHR_finetuning_gradientAccumulation.pt"
model_name_MAE="Residual_MultipleMultiHeadCrossAttention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart"
model_name_CLIP="Residual_MultipleMultiHeadCrossAttention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart_MSE_CLIPLoss"
vae_model_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(vae_model_path)

model = Residual_CrossAttention_UNet_superres(input_channels, output_channels, device).to(device)
vae_model = pipe.vae.to(device)
VAE_weight_path=os.path.join('models_run', VAE_model_name)
noise_schedule="cosine"
noise_steps=1000
Blur_radius=0.5
Degradation_type="DownBlur"
snapshot_path_MAE=os.path.join('models_run', model_name_MAE, 'weights', 'snapshot.pt')
snapshot_path_CLIP=os.path.join('models_run', model_name_CLIP, 'weights', 'snapshot.pt')

diffusion_MAE = Diffusion(
    noise_schedule=noise_schedule, model=model, vae_model=vae_model,
    snapshot_path=snapshot_path_MAE,
    VAE_weight_path=VAE_weight_path,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name_MAE, Degradation_type=Degradation_type,
    multiple_gpus=False, ema_smoothing=False)

diffusion_CLIP = Diffusion(
    noise_schedule=noise_schedule, model=model, vae_model=vae_model,
    snapshot_path=snapshot_path_CLIP,
    VAE_weight_path=VAE_weight_path,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name_CLIP, Degradation_type=Degradation_type,
    multiple_gpus=False, ema_smoothing=False)

transform_sr_bicubic = transforms.Compose([
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])
transform_lr = transforms.Compose([
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

img_name = 'patch_768_7680.png'
img = Image.open(os.path.join('up42_sentinel2_patches','test_original', img_name))
sr_bicubic = transform_sr_bicubic(img).unsqueeze(0).to(device)
lr_img = transform_lr(img).unsqueeze(0).to(device)
latent_lr_img, latent_sr_img, superres_img_MAE = diffusion_MAE.sample(n=1,model=model, lr_img=lr_img, generate_video=False)
latent_lr_img, latent_sr_img, superres_img_CLIP = diffusion_CLIP.sample(n=1,model=model, lr_img=lr_img, generate_video=False)
sr_SwinIR = Image.open(os.path.join("..", "SWINIR", "results", "swinir_real_sr_x4", img_name.replace(".png", "_SwinIR.png"))).convert("RGB")

fig, axs = plt.subplots(2,3, figsize=(10,10))
axs = axs.ravel()
axs[0].imshow(lr_img[0].permute(1,2,0).cpu())
axs[0].set_title('LR_img')
axs[1].imshow(img)
axs[1].set_title('HR_img')
axs[2].imshow(superres_img_MAE[0].permute(1,2,0).detach().cpu())
axs[2].set_title('SR(latent_MAE)')
axs[3].imshow(superres_img_CLIP[0].permute(1,2,0).detach().cpu())
axs[3].set_title('SR(latent_CLIP)')
axs[4].imshow(sr_bicubic[0].permute(1,2,0).detach().cpu())
axs[4].set_title('SR(bicubic)')
axs[5].imshow(sr_SwinIR)
axs[5].set_title('SR(SwinIR)')
plt.show()
# %% TO REMOVE
from generate_new_imgs.utils import dataset_maker, NPYFolderDataset
from generate_new_imgs.UNet_model_generation_CrossAttention import Residual_Attention_UNet_generation
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from torch import nn

def VAE_model_maker(device):
    vae_model_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(vae_model_path)
    vae_model = pipe.vae.to(device)
    return vae_model

def _load_snapshot_VAE(snapshot_path, model):
    '''
    This function loads the model state and the last epoch of training (so that we can restart the
    training at this point instead of restarting from 0) from a snapshot.
    It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
    it from the last snapshot.
    '''
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    model.load_state_dict(snapshot)

    print(f"Snapshot loaded from {snapshot_path}")

device='cuda'
snapshot_path = os.path.join('models_run','VAE_102flowers_finetuning_gradientAccumulation.pt')
vae_model = VAE_model_maker(device=device)
vae_model.eval()
vae_model = vae_model.to(device)
_load_snapshot_VAE(snapshot_path, vae_model)

image_size = 512
dataset_original_path = os.path.join("102flowers_dataset")
dataset_original = dataset_maker(image_size, dataset_original_path)
dataset_encoded_path = os.path.join("102flowers_dataset_VAE_encoded")
dataset_encoded = NPYFolderDataset(dataset_encoded_path)

for original_img_tuple, encoded_img_tuple in zip(dataset_original.samples, dataset_encoded.samples):
    if original_img_tuple[1] == 0:
        original_img = transforms.ToTensor()(Image.open(original_img_tuple[0])).unsqueeze(0).to('cuda')
        encoded_img = transforms.ToTensor()(np.load(encoded_img_tuple[0])).unsqueeze(0).to('cuda')/0.18215

        with torch.no_grad():
            encoded_original_img = vae_model.encode(original_img).latent_dist.sample()
            decoded_original_img = vae_model.decode(encoded_original_img).sample
            decoded_encoded_img = vae_model.decode(encoded_img).sample
        break
        fig, axs = plt.subplots(2,5, figsize=(10,10))
        axs = axs.ravel()

        axs[0].imshow(original_img[0].permute(1,2,0).detach().cpu())
        axs[1].imshow(encoded_img[0].permute(1,2,0).detach().cpu())
        axs[2].imshow(encoded_original_img[0].permute(1,2,0).detach().cpu())
        axs[3].imshow(decoded_original_img[0].permute(1,2,0).detach().cpu())
        axs[4].imshow(decoded_encoded_img[0].permute(1,2,0).detach().cpu())
        axs[5].hist(original_img.ravel().cpu())
        axs[6].hist(encoded_img.ravel().cpu())
        axs[7].hist(encoded_original_img.ravel().cpu())
        axs[8].hist(decoded_original_img.ravel().cpu())
        axs[9].hist(decoded_encoded_img.ravel().cpu())
        plt.show()
        break
# %%
