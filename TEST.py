#%% FROM NUMPY TO TORCH
import os
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

to_tensor = transforms.ToTensor()

sets = ["train_original", "val_original"]
img_types = ["hr_img", "lr_img"]

for _set in sets:
    for img_type in img_types:
        folder_path = os.path.join("up42_sentinel2_patches_VAE_encoded_numpy", _set, img_type)
        imgs_filenames = os.listdir(folder_path)
        output_folder_path = os.path.join("up42_sentinel2_patches_VAE_encoded",_set, img_type)
        os.makedirs(output_folder_path, exist_ok=True)
        if len(os.listdir(output_folder_path)) == 0:
            for img_filename in tqdm(imgs_filenames):
                img_path = os.path.join(folder_path, img_filename)
                img = to_tensor(np.load(img_path))
                img_output_path = os.path.join(output_folder_path, img_filename.replace("npy", "pt"))
                torch.save(img, img_output_path)
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
    stacked[:, :, i] = band
    stacked[:, :, i] = (band - band.min()) / (band.max() - band.min())

patches = patchify(
    stacked,
    (patch_size, patch_size, stacked.shape[2]),  # (H, W, C)
    step=step
)

# %% SUPER-RESOLUTION from array
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

def img_processing_from_numpy(numpy_array, image_size, degradation=True):
    # Ensure numpy array is uint8
    if numpy_array.dtype != np.uint8:
        numpy_array = (numpy_array * 255).clip(0, 255).astype(np.uint8)

    # Convert NumPy array to PIL Image
    if numpy_array.ndim == 2:  # grayscale
        img = Image.fromarray(numpy_array, mode='L')
    elif numpy_array.shape[2] == 3:  # RGB
        img = Image.fromarray(numpy_array, mode='RGB')
    else:
        raise ValueError("Unsupported image shape")

    to_tensor = transforms.ToTensor()

    if degradation:
        magnification_factor = 4
        blur_radius = 0.5

        # Resize (high quality image first)
        transform = transforms.Resize((image_size, image_size))
        img = transform(img)

        # Downsample
        downsample = transforms.Resize(
            (img.size[1] // magnification_factor,
             img.size[0] // magnification_factor),
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        img_down = downsample(img)

        # Blur
        img_down = img_down.filter(ImageFilter.GaussianBlur(blur_radius))

        # Convert to tensor
        img_down = to_tensor(img_down)

    else:
        img = transforms.Resize((image_size, image_size))(img)
        img_down = to_tensor(img)

    return img_down

img = img_processing_from_numpy(patches[0][10][0], image_size=image_size, degradation=True).to(device)
img_original = patches[0][10][0]

to_tensor = transforms.ToTensor()
img_original_tensor = to_tensor(img_original).unsqueeze(0).to(device)
latent_hr_img = diffusion.vae_model.encode(img_original_tensor).latent_dist.sample()

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
# %% SUPER-RESOLUTION from path
from train_diffusion_superres import Diffusion, UNet_model_maker, VAE_model_maker
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

device = "cuda"
model_name="Residual_MultipleMultiHeadCrossAttention_UNet_superres_magnification2_LRimgsize128_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart_MSELoss"
VAE_model_name="VAE_up42_LRandHR_finetuning_gradientAccumulation.pt"
snapshot_name="snapshot.pt"
UNet_type="Residual Cross Attention UNet"
input_channels=output_channels=4
snapshot_folder_path = os.path.join(os.curdir, 'models_run', model_name, 'weights')
image_size=256
noise_schedule="cosine"
noise_steps=1000
magnification_factor=2
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

def img_processing_from_path(img_path, image_size, magnification_factor, degradation=False):
    img = Image.open(img_path)
    to_tensor = transforms.ToTensor()
    if degradation:
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


img_folder_path = os.path.join("up42_sentinel2_patches","test_original")
output_folder_path = os.path.join("lr_vs_sr_comparison")
os.makedirs(output_folder_path, exist_ok=True)
for filename in os.listdir(img_folder_path)[30:31]:
    img_path = os.path.join(img_folder_path, filename)
    img = img_processing_from_path(img_path, image_size=image_size, magnification_factor=magnification_factor, degradation=True).to(device)
    img_original = np.array(Image.open(img_path))

    to_tensor = transforms.ToTensor()
    img_original_tensor = to_tensor(img_original).unsqueeze(0).to(device)
    latent_hr_img = diffusion.vae_model.encode(img_original_tensor).latent_dist.sample()

    fig, axs = plt.subplots(2,3, figsize=(15,15))
    axs = axs.ravel()
    latent_lr_img, latent_sr_img, superres_img = diffusion.sample(n=1,model=model, lr_img=img, generate_video=False)

    axs[0].imshow(img.permute(1,2,0).detach().cpu().numpy())
    axs[0].set_title('Low resolution image')
    axs[0].axis("off")
    axs[1].imshow(img_original)
    axs[1].set_title('High resolution image')
    axs[1].axis("off")
    axs[2].imshow(superres_img[0].permute(1,2,0).detach().cpu().numpy())
    axs[2].set_title('Super resolution image')
    axs[2].axis("off")

    axs[3].imshow(latent_lr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
    axs[3].set_title('Low resolution latent')
    axs[3].axis("off")
    axs[4].imshow(latent_hr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
    axs[4].set_title('High resolution latent')
    axs[4].axis("off")
    axs[5].imshow(latent_sr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
    axs[5].set_title('Super resolution latent')
    axs[5].axis("off")
    plt.savefig(os.path.join(output_folder_path,f"{filename}"), dpi=300, bbox_inches="tight")

    plt.show()
    


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

# %% SENTINEL 2 BIG IMAGE PROCESSING
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

meters_resolution = str(20)
blue_channel = rasterio.open(os.path.join("Napoli_sentinel2",f"R{meters_resolution}m",f"T33TVF_20250205T095231_B02_{meters_resolution}m.jp2")).read(1).astype(np.float32)[:,:, None]
green_channel = rasterio.open(os.path.join("Napoli_sentinel2",f"R{meters_resolution}m",f"T33TVF_20250205T095231_B03_{meters_resolution}m.jp2")).read(1).astype(np.float32)[:,:, None]
red_channel = rasterio.open(os.path.join("Napoli_sentinel2",f"R{meters_resolution}m",f"T33TVF_20250205T095231_B04_{meters_resolution}m.jp2")).read(1).astype(np.float32)[:,:, None]
# scl_channel = rasterio.open(os.path.join("Napoli_sentinel2",f"R{meters_resolution}m",f"T33TVF_20250205T095231_SCL_{meters_resolution}m.jp2")).read(1).astype(np.float32)[:,:, None]
# cloud_mask = (scl_channel == 3) | (scl_channel == 8) | (scl_channel == 9) | (scl_channel == 10) | (scl_channel == 11)

SCALE_FACTOR = 10000.0  # for L2A (use 1E4 for L1C)


# Convert DN to reflectance
blue_channel = np.clip(blue_channel / SCALE_FACTOR, 0,1)
green_channel = np.clip(green_channel / SCALE_FACTOR, 0,1)
red_channel = np.clip(red_channel / SCALE_FACTOR, 0,1)

# blue_channel[cloud_mask] = np.nan
# red_channel[cloud_mask] = np.nan
# green_channel[cloud_mask] = np.nan

rgb = np.concatenate([red_channel, green_channel, blue_channel], axis=2)*255

rgb = rgb.astype(np.uint8)
Image.fromarray(rgb).save(f'rgb_{meters_resolution}m.png')

plt.imshow(rgb)
plt.title("Sentinel-2 RGB")
plt.show()
# %%
