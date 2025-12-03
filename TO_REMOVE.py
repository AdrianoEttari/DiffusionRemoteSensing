#%% SAR TO NDVI
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from tqdm import tqdm

class VAE_model_wrapped(nn.Module):
    def __init__(self, vae_model, in_channels, freeze_vae_params=False):
        super(VAE_model_wrapped, self).__init__()
        self.vae_model = vae_model
        self.in_channels = in_channels
        self.SCALE = 0.18215
        self.conv_start = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1)
        self.conv_end = nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1)
        if freeze_vae_params: #if True, the vae parameters are frozen and just the conv layer is trained
            for param in self.vae_model.parameters():
                param.requires_grad = False
    def encode(self, x):
        x = self.conv_start(x)
        latents = self.vae_model.encode(x).latent_dist.sample() * self.SCALE
        return latents

    def decode(self, latents):
        x = self.vae_model.decode(latents).sample / self.SCALE
        return self.conv_end(x)
    

def VAE_model_maker(device, freeze_vae_params):
    vae_model_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(vae_model_path)
    vae_model = pipe.vae.to(device)
    vae_model = VAE_model_wrapped(vae_model, in_channels=1, freeze_vae_params=freeze_vae_params).to(device)
    return vae_model


device="cuda"
vae_model = VAE_model_maker(device, True)
snapshot_path = "models_run\VAE_SAR_TO_NDVI_finetuning_gradientAccumulation.pt"
snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
vae_model.load_state_dict(snapshot)
vae_model.eval()

# %%  SAR TO NDVI
sar_train_path = os.path.join("SAR_TO_NDVI_dataset","train","sar")
opt_train_path = os.path.join("SAR_TO_NDVI_dataset","train","opt")

def compute_global_min_max(root_dir):
    data_path = f'../{root_dir}' if os.path.exists(f'../{root_dir}') else root_dir

    global_min = float('inf')
    global_max = float('-inf')

    for fname in tqdm(os.listdir(data_path)):
        if fname.endswith(".npy"):
            arr = np.load(os.path.join(data_path, fname))
            global_min = min(global_min, arr.min())
            global_max = max(global_max, arr.max())
        elif fname.endswith(".pt"):
            arr = vae_model.encode(torch.load(os.path.join(data_path, fname))[0,:,:].unsqueeze(0).unsqueeze(0).to(device))
            global_min = min(global_min, arr.min())
            global_max = max(global_max, arr.max())
        else:
            raise ValueError(f"The files are not in numpy type. {fname}")
    return global_min, global_max

sar_min, sar_max = compute_global_min_max(sar_train_path)
print(sar_min, sar_max)
opt_min, opt_max = compute_global_min_max(opt_train_path)
print(opt_min, opt_max)
# %% SAR TO NDVI
sar_encoded_train_path = os.path.join("SAR_TO_NDVI_dataset_VAE_encoded","train","sar")
opt_encoded_train_path = os.path.join("SAR_TO_NDVI_dataset_VAE_encoded","train","opt")

def compute_global_min_max(root_dir):
    data_path = f'../{root_dir}' if os.path.exists(f'../{root_dir}') else root_dir

    global_min = float('inf')
    global_max = float('-inf')

    for fname in tqdm(os.listdir(data_path)):
        if fname.endswith(".npy"):
            arr = np.load(os.path.join(data_path, fname))
            global_min = min(global_min, arr.min())
            global_max = max(global_max, arr.max())
        elif fname.endswith(".pt"):
            arr = torch.load(os.path.join(data_path, fname))
            global_min = min(global_min, arr.min())
            global_max = max(global_max, arr.max())
        else:
            raise ValueError(f"The files are not in numpy type. {fname}")
    return global_min, global_max

sar_min, sar_max = compute_global_min_max(sar_encoded_train_path)
print(sar_min, sar_max)
opt_min, opt_max = compute_global_min_max(opt_encoded_train_path)
print(opt_min, opt_max)
# %% SAR TO NDVI
from utils import get_data_SAR_TO_NDVI
train_path = os.path.join("SAR_TO_NDVI_dataset", "train")
train_dataset = get_data_SAR_TO_NDVI(train_path,SAR_channels=1,transform=None)

global_min_sar = float('inf')
global_max_sar = float('-inf')
global_min_opt = float('inf')
global_max_opt = float('-inf')

for sar_img, opt_img in tqdm(train_dataset):
    sar_img = sar_img.unsqueeze(0).to(device)
    opt_img = opt_img.unsqueeze(0).to(device)
    sar_img = vae_model.encode(sar_img)
    opt_img = vae_model.encode(opt_img)
    global_min_sar = min(global_min_sar, sar_img.min())
    global_max_sar = max(global_max_sar, sar_img.max())
    global_min_opt = min(global_min_opt, opt_img.min())
    global_max_opt = max(global_max_opt, opt_img.max())
print(global_min_sar, global_max_sar)
print(global_min_opt, global_max_opt)
# %% SAR TO NDVI
from utils import compute_global_min_max
import os

train_path_sar = os.path.join("SAR_TO_NDVI_dataset_VAE_encoded", "train", "sar")
global_min_sar, global_max_sar = compute_global_min_max(train_path_sar)

train_path_opt = os.path.join("SAR_TO_NDVI_dataset_VAE_encoded", "train", "opt")
global_min_opt, global_max_opt = compute_global_min_max(train_path_opt)
print(global_min_sar, global_max_sar, global_min_opt, global_max_opt)

val_path_sar = os.path.join("SAR_TO_NDVI_dataset_VAE_encoded", "val", "sar")
global_min_sar, global_max_sar = compute_global_min_max(val_path_sar)

val_path_opt = os.path.join("SAR_TO_NDVI_dataset_VAE_encoded", "val", "opt")
global_min_opt, global_max_opt = compute_global_min_max(val_path_opt)

print(global_min_sar, global_max_sar, global_min_opt, global_max_opt)
# %% SUPER-RESOLUTION
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from tqdm import tqdm

def VAE_model_maker(device):
    vae_model_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(vae_model_path)
    vae_model = pipe.vae.to(device)
    return vae_model


device="cuda"
vae_model = VAE_model_maker(device)
snapshot_path = "models_run\VAE_up42_LRandHR_finetuning_gradientAccumulation.pt"
snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
vae_model.load_state_dict(snapshot)
vae_model.eval()
# %% compute_global_min_max
from torchvision import transforms

train_path = os.path.join("up42_sentinel2_patches","train_original")

def compute_global_min_max(root_dir):
    data_path = f'../{root_dir}' if os.path.exists(f'../{root_dir}') else root_dir
    SCALE = 0.18125
    global_min = float('inf')
    global_max = float('-inf')

    for fname in tqdm(os.listdir(data_path)):
        arr = Image.open(os.path.join(data_path, fname))
        arr = transforms.ToTensor()(arr).unsqueeze(0).to(device)
        arr = vae_model.encode(arr).latent_dist.sample() * SCALE
        global_min = min(global_min, arr.min())
        global_max = max(global_max, arr.max())
    return global_min, global_max

_min, _max = compute_global_min_max(train_path)
print(_min, _max)
    

# %%
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
sr_img = np.array(Image.open("rgb_60m_SR.png"))
lr_img = np.array(Image.open("rgb_60m.png"))

min_x = 1250
max_x = 1500
min_y = 250
max_y = 500

fig, axs = plt.subplots(1,2,figsize=(15,30))

lr_img_to_plot = lr_img[min_x:max_x, min_y:max_y,:]
sr_img_to_plot = sr_img[min_x*2:max_x*2, min_y*2:max_y*2,:]
axs[0].imshow(lr_img_to_plot)
axs[0].axis("off")
axs[1].imshow(sr_img_to_plot)
axs[1].axis("off")
plt.savefig("lr_vs_sr_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

