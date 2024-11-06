#%%
import os
from diffusers import StableDiffusionPipeline
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch
import numpy as np
from tqdm import tqdm
from utils import get_data_superres
import matplotlib.pyplot as plt
from PIL import Image

VAE_SAVE_PATH = os.path.join('models_run', 'VAE_finetuning_HR.pt')

def load_super_res_VAE(model_path=VAE_SAVE_PATH, device='cuda'):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    vae = pipe.vae
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}...")
        vae = vae.load_state_dict(torch.load(model_path))
    else:
        print("Loading pretrained model...")

    vae = vae.to(device)
    return vae

#%% FINE-TUNING

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[3, 8, 15], device='cuda'):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1').features.to(device).eval()
        self.layers = feature_layers
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(feature_layers) + 1)])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, sr_images, hr_images):
        sr_images = self._preprocess_for_vgg(sr_images)
        hr_images = self._preprocess_for_vgg(hr_images)
        sr_features = self._extract_features(sr_images)
        hr_features = self._extract_features(hr_images)
        loss = 0
        for sr_feat, hr_feat in zip(sr_features, hr_features):
            loss += nn.functional.mse_loss(sr_feat, hr_feat)
        return loss

    def _extract_features(self, images):
        features = []
        x = images
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

    def _preprocess_for_vgg(self, images):
        # Assumes input images are in range [0, 1]
        images = (images - torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return images

# Fine-tune function with Perceptual Loss
def fine_tune_super_resolution(vae,
                                data_path,
                                magnification_factor,
                                Blur_radius,
                                image_size,
                                epochs=5,
                                batch_size=4,
                                learning_rate=1e-5,
                                save_path=VAE_SAVE_PATH,
                                device='cuda'):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),interpolation=transforms.InterpolationMode.BICUBIC),
    ])

    dataset = get_data_superres(data_path, magnification_factor, Blur_radius, False, 'PIL', transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)
    perceptual_loss_fn = PerceptualLoss(device=device)
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for lr_images, hr_images in tqdm(dataloader):
            lr_images = [transform(img) for img in lr_images]
            lr_images = torch.stack([img for img in lr_images]).to(device).to(torch.float32)
            hr_images = torch.stack([img  for img in hr_images]).to(device).to(torch.float32)

            # latents = vae.encode(lr_images).latent_dist.sample()
            # reconstructed_images = vae.decode(latents).sample
            # loss = perceptual_loss_fn(reconstructed_images, lr_images)

            latents = vae.encode(hr_images).latent_dist.sample()
            reconstructed_images = vae.decode(latents).sample
            loss = perceptual_loss_fn(reconstructed_images, hr_images)

            # latents = vae.encode(lr_images).latent_dist.sample()
            # reconstructed_images = vae.decode(latents).sample
            # loss = perceptual_loss_fn(reconstructed_images, hr_images)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(vae.state_dict(), save_path)
    print(f"Fine-tuned model saved at {save_path}")

    vae.eval()
    return vae

# Usage
device = 'cuda'
vae = load_super_res_VAE(device=device)
data_path = os.path.join('celebA_100k','train_original')
# data_path = os.path.join('celebA_10k','train_original')
magnification_factor = 4
Blur_radius = 0.5
image_size = 192
fine_tuned_pipe = fine_tune_super_resolution(vae,
                                            data_path,
                                            magnification_factor,
                                            Blur_radius,
                                            image_size,
                                            epochs=10,
                                            batch_size=4,
                                            learning_rate=1e-5,
                                            device=device)

#%% TESTING
def lr_image_preprocessing(image_path, lr_image_size: int, hr_image_size: int):
    image = Image.open(image_path)

    transform_resize_lr_size = transforms.Resize((lr_image_size,lr_image_size), transforms.InterpolationMode.BICUBIC)
    transform_resize_hr_size = transforms.Resize((hr_image_size,hr_image_size), transforms.InterpolationMode.BICUBIC)

    image = transforms.ToTensor()(image).unsqueeze(0).to(device)

    lr_image = transform_resize_hr_size(transform_resize_lr_size(image))
    return lr_image
    

device='mps'
# image_path = os.path.join('celebA_100k','test_original','000114.jpg')
image_path = os.path.join('celebA_10k','test_original','000114.jpg')
image = Image.open(image_path)
image = transforms.ToTensor()(image).unsqueeze(0).to(device)
hr_image_size = 192
magnification_factor = 4
lr_image_size = hr_image_size//magnification_factor

fine_tuned_vae = load_super_res_VAE(VAE_SAVE_PATH,device=device)
transform_resize_192 = transforms.Resize((hr_image_size,hr_image_size), transforms.InterpolationMode.BICUBIC)

lr_image = lr_image_preprocessing(image_path, lr_image_size, hr_image_size)
hr_image = transform_resize_192(image)

# latents = fine_tuned_vae.encode(lr_image).latent_dist.sample()
latents = fine_tuned_vae.encode(hr_image).latent_dist.sample()
reconstruction_images = fine_tuned_vae.decode(latents).sample



fig, axs = plt.subplots(1,3, figsize=(10,5))
axs = axs.ravel()

axs[0].imshow(lr_image[0].permute(1,2,0).detach().cpu())
axs[0].set_title("Low Resolution Image")
axs[0].axis('off')
axs[1].imshow(reconstruction_images[0].permute(1,2,0).detach().cpu())
axs[1].set_title("Reconstruction Image")
axs[1].axis('off')
axs[2].imshow(hr_image[0].permute(1,2,0).detach().cpu())
axs[2].set_title("Original Image")
axs[2].axis('off')
plt.savefig('Reconstruction Image HR.png')


# %%
