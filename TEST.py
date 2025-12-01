#%% SENTINEL 2 LOADING AND SHOWING
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

meters_resolution = str(20)
folder_path = os.path.join("Napoli_sentinel2",f"R{meters_resolution}m")
filenames_to_read = [f"T33TVF_20250205T095231_B02_{meters_resolution}m.jp2", f"T33TVF_20250205T095231_B03_{meters_resolution}m.jp2", f"T33TVF_20250205T095231_B04_{meters_resolution}m.jp2"]

images_arrays = []
for filename in filenames_to_read:
    file_path = os.path.join(folder_path, filename)
    img = Image.open(file_path)
    img = np.array(img)
    images_arrays.append(img)

stacked = np.stack(images_arrays, axis=-1).astype(np.float32)

def plot_sentinel2_img(img, p2=None, p98=None):
    """
    TO BE USED JUST FOR PLOTTING!!!
    """
    if p2 is None or p98 is None:
        p2, p98 = np.percentile(img, (2, 98))
    img = np.clip((img - p2) / (p98 - p2), 0, 1)
    img = img**0.8 # gamma correction
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")

plot_sentinel2_img(stacked)
#%% patchify Naples
from patchify import patchify

patch_size = 256   # or 32, 128, etc.
step = 256         # use same as patch_size for non-overlapping

# stacked: (H, W, 3) float32
stacked = np.stack(images_arrays, axis=-1).astype(np.float32)

# Per-band min-max normalization to [0,1]
for i in range(stacked.shape[-1]):
    band = stacked[:, :, i]
    stacked[:, :, i] = (band - band.min()) / (band.max() - band.min())

patches = patchify(
    stacked,
    (patch_size, patch_size, stacked.shape[2]),  # (H, W, C)
    step=step
)


# %% SUPER-RESOLUTION
from train_diffusion_superres import Diffusion, UNet_model_maker, VAE_model_maker
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

device = "cuda"
model_name="Residual_MultipleMultiHeadCrossAttention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart_MSE_CLIPLoss"
VAE_model_name="VAE_up42_LRandHR_finetuning_gradientAccumulation.pt"
snapshot_name="snapshot.pt"
UNet_type="Residual Cross Attention UNet"
input_channels=output_channels=4
snapshot_folder_path = os.path.join(os.curdir, 'models_run', model_name, 'weights')
image_size=256
noise_schedule="cosine"
noise_steps=1000
magnification_factor=4
Degradation_type="DownBlur"
Blur_radius=0.5
generate_video=False
VAE_weight_path = os.path.join('models_run', VAE_model_name)

model = UNet_model_maker(UNet_type, input_channels, output_channels, device, image_size)
print("Num params: ", sum(p.numel() for p in model.parameters()))

vae_model = VAE_model_maker(device)

snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

diffusion = Diffusion(
    noise_schedule=noise_schedule, model=model, vae_model=vae_model,
    snapshot_path=snapshot_path,
    VAE_weight_path=VAE_weight_path,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name, Degradation_type=Degradation_type,
    multiple_gpus=False, ema_smoothing=False)

def img_processing(img_path, image_size, degradation=False):
    img = Image.open(img_path)
    to_tensor = transforms.ToTensor()
    if degradation:
        magnification_factor = 4
        blur_radius = 0.5
        transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        ])

        img = transform(img)
        downsample = transforms.Resize((img.size[0] // magnification_factor, img.size[1] // magnification_factor),
                                interpolation=transforms.InterpolationMode.BICUBIC)
        
        try:
            img_down = downsample(img)
        except:
            img_down = downsample(img.to('cpu')).to(img.device)

        img_down = img_down.filter(ImageFilter.GaussianBlur(blur_radius))
        img_down = to_tensor(img_down)
        # img = to_tensor(img)
    else:
        img_down = to_tensor(img)
    return img_down

img_path = os.path.join("up42_sentinel2_patches","test_original","patch_256_7424.png")

img = img_processing(img_path, image_size=image_size, degradation=True).to(device)
img_original = np.array(Image.open(img_path))

to_tensor = transforms.ToTensor()
img_original_tensor = to_tensor(img_original).unsqueeze(0).to(device)
latent_hr_img = diffusion.vae_model.encode(img_original_tensor).latent_dist.sample()
######### SAMPLING ##########
fig, axs = plt.subplots(2,3, figsize=(15,15))
axs = axs.ravel()
latent_lr_img, latent_sr_img, superres_img = diffusion.sample(n=1,model=model, lr_img=img, generate_video=False)

axs[0].imshow(img.permute(1,2,0).detach().cpu().numpy())
axs[0].set_title('Low resolution image')
axs[1].imshow(img_original)
axs[1].set_title('High resolution image')
axs[2].imshow(superres_img[0].permute(1,2,0).detach().cpu().numpy())
axs[2].set_title('Super resolution image')

axs[3].imshow(latent_lr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
axs[3].set_title('Low resolution latent')
axs[4].imshow(latent_hr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
axs[4].set_title('High resolution latent')
axs[5].imshow(latent_sr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
axs[5].set_title('Super resolution latent')
plt.show()
# plt.savefig(os.path.join(os.getcwd(), 'models_run', model_name, 'results', 'superres_results.png'))

# %% EXAMPLE VAE ON up42
from train_diffusion_superres import Diffusion, UNet_model_maker, VAE_model_maker
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

device = "cuda"
model_name="Residual_MultipleMultiHeadCrossAttention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart_MSE_CLIPLoss"
VAE_model_name="VAE_up42_LRandHR_finetuning_gradientAccumulation.pt"
snapshot_name="snapshot.pt"
UNet_type="Residual Cross Attention UNet"
input_channels=output_channels=4
snapshot_folder_path = os.path.join(os.curdir, 'models_run', model_name, 'weights')
image_size=256
noise_schedule="cosine"
noise_steps=1000
magnification_factor=4
Degradation_type="DownBlur"
Blur_radius=0.5
generate_video=False
VAE_weight_path = os.path.join('models_run', VAE_model_name)

model = UNet_model_maker(UNet_type, input_channels, output_channels, device, image_size)
print("Num params: ", sum(p.numel() for p in model.parameters()))

vae_model = VAE_model_maker(device)

snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

diffusion = Diffusion(
    noise_schedule=noise_schedule, model=model, vae_model=vae_model,
    snapshot_path=snapshot_path,
    VAE_weight_path=VAE_weight_path,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
    magnification_factor=magnification_factor,device=device,
    image_size=image_size, model_name=model_name, Degradation_type=Degradation_type,
    multiple_gpus=False, ema_smoothing=False)

img_path = os.path.join('up42_sentinel2_patches','test_original','patch_0_4864.png')
img = Image.open(img_path).resize((image_size, image_size))
lr_img = img.resize((image_size//magnification_factor, image_size//magnification_factor))
transform = transforms.ToTensor()
hr_img = transform(img).unsqueeze(0).to(device)
lr_img = transform(lr_img).unsqueeze(0).to(device)

latents = diffusion.vae_model.encode(hr_img).latent_dist.sample()
reconstructed = diffusion.vae_model.decode(latents).sample

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(lr_img[0].permute(1, 2, 0).cpu())
axs[0].set_title('Low Resolution Image')
axs[1].imshow(hr_img[0].permute(1, 2, 0).cpu())
axs[1].set_title('High Resolution Image')
axs[2].imshow(reconstructed[0].permute(1, 2, 0).cpu().detach().numpy())
axs[2].set_title('Reconstructed Image')
plt.show()