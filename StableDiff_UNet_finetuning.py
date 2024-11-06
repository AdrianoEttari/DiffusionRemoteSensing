# # %%
from diffusers import StableDiffusionPipeline
from transformers import ViTModel, ViTFeatureExtractor
import torch

# Load Stable Diffusion pipeline
model_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_path)
device = 'mps'
# Extract components for full-model training
unet = pipe.unet.to(device)
vae = pipe.vae.to(device)
noise_scheduler = pipe.scheduler  # e.g., DDIMScheduler or PNDMScheduler

# Load an image encoder for image conditioning (e.g., ViT)
image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

#%%
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import get_data_superres
from tqdm import tqdm

# Initialize pipeline and optimizer
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-5)

# Set hyperparameters and transformations
num_epochs = 10
batch_size = 4
image_size = 192
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
])

# Set loss function and scheduler configuration
loss_function = torch.nn.MSELoss()
noise_steps = pipe.scheduler.config.num_train_timesteps

# Load dataset
data_path = os.path.join('celebA_10k', 'train_original')
magnification_factor = 4
Blur_radius = 0.5
dataset = get_data_superres(data_path, magnification_factor, Blur_radius, False, 'PIL', transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fine-tuning loop
for epoch in range(num_epochs):
    for lr_image, hr_image in tqdm(dataloader):
        # Move data to device
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)

        optimizer.zero_grad()

        # Sample a random timestep
        timestep = torch.randint(low=1, high=noise_steps, size=(1,)).to(device)

        # Resize and condition on low-resolution image embeddings
        transform_resize_224 = transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC)
        lr_image_resized = transform_resize_224(lr_image.to("cpu")).to(device)
        conditioning_embedding = image_encoder(lr_image_resized).last_hidden_state.to(device)

        # Step 1: Encode high-resolution image into latent space
        latent_hr_image = pipe.vae.encode(hr_image).latent_dist.sample() * 0.18215 # 0.18215 (is the std of the prior) is used to scale the latent appropriately in the UNet

        # Step 2: Add noise to the latent image at the given timestep
        noise = torch.randn_like(latent_hr_image).to(device)
        noisy_latent_image = pipe.scheduler.add_noise(latent_hr_image, noise, timestep)

        # Step 3: Pass noisy latent image to UNet
        noise_pred = pipe.unet(noisy_latent_image, timestep, conditioning_embedding).sample

        # Calculate loss and update weights
        loss = loss_function(noise_pred, noise)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")