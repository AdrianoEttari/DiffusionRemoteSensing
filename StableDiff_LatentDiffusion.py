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
from transformers import ViTModel, ViTFeatureExtractor

class LatentDiffusion_superres:
    def __init__(self,
                VAE_weight_path, 
                Diffusion_weight_path,
                device='cuda') -> None:
        
        self.VAE_weight_path = VAE_weight_path
        self.Diffusion_weight_path = Diffusion_weight_path
        self.device = device

        model_path = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_path).to(self.device)
        self.pipe = pipe

        if os.path.exists(self.VAE_weight_path):
            print(f"Loading fine-tuned VAE model from {self.VAE_weight_path}...")
            self.pipe.vae = self.pipe.vae.load_state_dict(torch.load(self.VAE_weight_path))
        
        if os.path.exists(self.Diffusion_weight_path):
            print(f"Loading fine-tuned Diffusion model from {self.Diffusion_weight_path}...")
            self.pipe.unet = pipe.unet.load_state_dict(torch.load(self.Diffusion_weight_path))
        
    def fine_tuning_VAE(self,
                        data_path,
                        magnification_factor,
                        Blur_radius,
                        image_size,
                        epochs,
                        batch_size,
                        learning_rate):
        
        print("Fine-tuning VAE...")
        device = self.device
        vae = self.pipe.vae
        save_path = self.VAE_weight_path

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
                hr_images = torch.stack([img for img in hr_images]).to(device).to(torch.float32)

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
        print(f"Fine-tuned VAE model saved at {save_path}")

        vae.eval()
        return vae

    def fine_tuning_Diffusion(self,
                            data_path,
                            magnification_factor,
                            Blur_radius,
                            image_size,
                            epochs,
                            batch_size,
                            learning_rate):
        
        print("Fine-tuning Diffusion...")
        pipe = self.pipe
        unet = self.pipe.unet
        device = self.device
        save_path = self.Diffusion_weight_path
        self.pipe.vae.eval()

        optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

        image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        # feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        ])

        # Set loss function and scheduler configuration
        loss_function = torch.nn.MSELoss()
        noise_steps = pipe.scheduler.config.num_train_timesteps

        # Load dataset
        dataset = get_data_superres(data_path, magnification_factor, Blur_radius, False, 'PIL', transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Fine-tuning loop
        unet.train()
        for epoch in range(epochs):
            total_loss = 0
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
                noise_pred = unet(noisy_latent_image, timestep, conditioning_embedding).sample

                # Calculate loss and update weights
                loss = loss_function(noise_pred, noise)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(unet.state_dict(), save_path)
        print(f"Fine-tuned Diffusion model saved at {save_path}")

        unet.eval()
        return unet

    def sample_superres(self, lr_image, num_inference_steps,):

        self.pipe.unet.eval() 
        self.pipe.vae.eval()
        device = self.device
        image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        self.pipe.scheduler.set_timesteps(num_inference_steps) # Set the number of inference steps. Notice that self.pipe.scheduler.timesteps is not from num_inference_steps to 0, but is a random sample of num_inference_steps timesteps from the range [1, self.pipe.scheduler.config.num_train_timesteps]

        with torch.no_grad():
            # Step 1: Start with random noise in the latent space
            latent_shape = (1, 4, 24, 24)  # CHANGE ACCORDING TO YOUR LATENT SPACE DIMENSIONS
            latent_sample = torch.randn(latent_shape).to(device)

            # Step 2: Perform iterative denoising

            for timestep in tqdm(self.pipe.scheduler.timesteps):
                timestep = timestep.to(device)

                # Compute noise prediction
                with torch.no_grad():
                    # Optionally, provide conditioning embeddings if needed
                    transform_resize_224 = transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC)
                    lr_image_resized = transform_resize_224(lr_image.to("cpu")).to(device)
                    conditioning_embedding = image_encoder(lr_image_resized).last_hidden_state.to(device)
                    noise_pred = self.pipe.unet(latent_sample, timestep, conditioning_embedding).sample
                # Update latent sample with the scheduler step
                latent_sample = self.pipe.scheduler.step(noise_pred, timestep, latent_sample).prev_sample # The scheduler’s step() method takes the noisy residual, timestep, and input and it predicts the image at the previous timestep
                # latent_sample = self.pipe.scheduler.step(noise_pred.to('cpu'), scaled_timestep.to('cpu'), latent_sample.to('cpu')).prev_sample

            # Step 3: Decode latent to image space using the VAE decoder
            generated_image = self.pipe.vae.decode(latent_sample / 0.18215).sample  # Scale by 1/0.18215 as done during training

        # The `generated_image` is now in the range typically [-1, 1] or [0, 1]. You may need to convert it to display/save.
        generated_image = (generated_image + 1) / 2  # Rescale to [0, 1]
        # generated_image = generated_image.clamp(0, 1)  # Ensure valid range

        return generated_image

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
    

if __name__ == "__main__":
    VAE_weight_path = os.path.join('models_run', 'VAE_finetuning')
    Diffusion_weight_path = os.path.join('models_run', 'Diffusion_finetuning')
    device = 'mps'
    latent_diff_model = LatentDiffusion_superres(VAE_weight_path=VAE_weight_path,
                                                Diffusion_weight_path=Diffusion_weight_path,
                                                device=device)
    
    # data_path = os.path.join('celebA_10k', 'train_original')
    # magnification_factor = 4
    # Blur_radius = 0.5
    # image_size = 192
    # epochs = 10
    # batch_size = 4
    # learning_rate = 1e-5
    # latent_diff_model.fine_tuning_VAE(data_path=data_path,
    #                                 magnification_factor=magnification_factor,
    #                                 Blur_radius=Blur_radius,
    #                                 image_size=image_size,
    #                                 epochs=epochs,
    #                                 batch_size=batch_size,
    #                                 learning_rate=learning_rate)
    lr_image = Image.open('celebA_10k/test_original/000100.jpg')
    lr_image = transforms.ToTensor()(lr_image).unsqueeze(0).to(device)
    super_res_image = latent_diff_model.sample_superres(lr_image, num_inference_steps=100)
    plt.imshow(super_res_image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.savefig('super_res_image.png')

    

