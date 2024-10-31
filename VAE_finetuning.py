import os
from diffusers import StableDiffusionPipeline
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
from tqdm import tqdm
from utils import get_data_superres

# Save and load paths
MODEL_SAVE_PATH = os.path.join('models_run', 'VAE_finetining')

# Load the Stable Diffusion model with pretrained weights, or load fine-tuned weights if available
def load_super_res_pipeline(model_path=MODEL_SAVE_PATH):
    if os.path.exists(model_path):
        # Load the fine-tuned weights if available
        print(f"Loading fine-tuned model from {model_path}...")
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    else:
        # Load pretrained Stable Diffusion pipeline
        print("Loading pretrained model...")
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe

# Fine-tune the model for super-resolution on satellite images
def fine_tune_super_resolution(pipe, data_path, magnification_factor, Blur_radius, image_size, epochs=5, batch_size=4, learning_rate=1e-5, save_path=MODEL_SAVE_PATH):
    # Prepare Dataset and Dataloader
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    dataset = get_data_superres(data_path, magnification_factor, Blur_radius, False, 'PIL', transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = optim.AdamW(pipe.vae.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # Mean Squared Error for pixel-wise loss

    pipe.vae.train()
    print("Num params: ", sum(p.numel() for p in pipe.vae.parameters()))
    # Training Loop
    for epoch in range(epochs):
        total_loss = 0
        for lr_images, hr_images in tqdm(dataloader):
            transform_resize = transforms.Resize((192,192), transforms.InterpolationMode.BICUBIC)
            lr_images = [transform_resize(img) for img in lr_images]
            lr_images = torch.stack([torch.tensor(np.array(img)).float()  for img in lr_images]).to("cuda").to(torch.float32)
            hr_images = torch.stack([torch.tensor(np.array(img)).float()  for img in hr_images]).to("cuda").to(torch.float32)
            
            # Encode LR image to latent space using the VAE
            latents = pipe.vae.encode(lr_images).latent_dist.sample()
            
            # Decode to high-res space
            sr_images = pipe.vae.decode(latents).sample
            loss = loss_fn(sr_images, hr_images)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pipe.save_pretrained(save_path)
    print(f"Fine-tuned model saved at {save_path}")

    # Set model components back to evaluation mode
    pipe.vae.eval()
    return pipe

# Usage
pipe = load_super_res_pipeline()
data_path = os.path.join('celebA_100k','train_original')
magnification_factor = 4
Blur_radius = 0.5
image_size = 192
fine_tuned_pipe = fine_tune_super_resolution(pipe, data_path, magnification_factor, Blur_radius, image_size, epochs=10, batch_size=4, learning_rate=1e-5)
