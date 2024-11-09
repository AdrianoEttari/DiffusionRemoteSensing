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


class LatentDiffusion_superres:
    def __init__(self,
                VAE_weight_path, 
                Diffusion_weight_path,
                device,
                multiple_gpus=False) -> None:
        
        self.VAE_weight_path = VAE_weight_path
        self.Diffusion_weight_path = Diffusion_weight_path
        self.multiple_gpus = multiple_gpus
        self.device = device
    
        model_path = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_path).to(self.device)
        self.pipe = pipe
        
        if self.multiple_gpus:
            self.pipe.vae = DDP(self.pipe.vae, device_ids=[self.device])
            self.pipe.unet = DDP(self.pipe.unet, device_ids=[self.device])
        
        if os.path.exists(self.VAE_weight_path):
            print(f"Loading fine-tuned VAE model from {self.VAE_weight_path}...")
            # snapshot = torch.load(self.VAE_weight_path, map_location=self.device, weights_only=True)
            # self.pipe.vae.load_state_dict(snapshot)
            self._load_snapshot(self.VAE_weight_path, self.pipe.vae)
        
        if os.path.exists(self.Diffusion_weight_path):
            print(f"Loading fine-tuned Diffusion model from {self.Diffusion_weight_path}...")
            # snapshot = torch.load(self.Diffusion_weight_path, map_location=self.device, weights_only=True)
            # self.pipe.unet.load_state_dict(snapshot)
            self._load_snapshot(self.Diffusion_weight_path, self.pipe.unet)
    
    def _load_snapshot(self, snapshot_path, model):
        '''
        This function loads the model state and the last epoch of training (so that we can restart the
        training at this point instead of restarting from 0) from a snapshot.
        It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
        it from the last snapshot.
        '''
        if self.multiple_gpus:
            from collections import OrderedDict

            snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=True)
            model_state = OrderedDict((key.replace('module.', ''), value) for key, value in snapshot.items())
            model.module.load_state_dict(model_state)
            model.module.to(self.device)
        else:
            snapshot = torch.load(snapshot_path, map_location=self.device, weights_only=True)
            model.load_state_dict(snapshot)

        print(f"Snapshot loaded from {snapshot_path}")

    def _save_snapshot(self, model, snapshot_path):
        '''
        This function loads the model state and the current epoch from a snapshot.
        It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
        it from the last snapshot.

        Input:
            model: the model to save

        Output:
            None
        '''
        if self.multiple_gpus:
            snapshot = model.module.state_dict()
        else:
            snapshot = model.state_dict()
        torch.save(snapshot, snapshot_path)
        print(f"Snapshot saved at {snapshot_path}")

    def fine_tuning_VAE(self,
                        dataloader,
                        image_size,
                        epochs,
                        learning_rate):

        print("Fine-tuning VAE...")
        device = self.device
        vae = self.pipe.vae
        save_path = self.VAE_weight_path

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size),interpolation=transforms.InterpolationMode.BICUBIC),
        ])

        optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)

        perceptual_loss_fn = PerceptualLoss(device=device)
        mse_loss_fn = torch.nn.MSELoss()
        loss_fn = CombinedLoss(perceptual_loss_fn, mse_loss_fn, alpha=0.5, device=device)

        vae.train()
        for epoch in range(epochs):
            total_loss = 0
            for lr_images, hr_images in tqdm(dataloader):
                lr_images = [transform(img) for img in lr_images]
                lr_images = torch.stack([img for img in lr_images]).to(device).to(torch.float32)
                hr_images = torch.stack([img for img in hr_images]).to(device).to(torch.float32)

                # latents = vae.encode(lr_images).latent_dist.sample()
                # reconstructed_images = vae.decode(latents).sample
                # loss = loss_fn(reconstructed_images, lr_images)

                if self.multiple_gpus:
                    latents = vae.module.encode(hr_images).latent_dist.sample()
                    reconstructed_images = vae.module.decode(latents).sample
                else:
                    latents = vae.encode(hr_images).latent_dist.sample()
                    reconstructed_images = vae.decode(latents).sample
                loss = loss_fn(reconstructed_images, hr_images)

                # latents = vae.encode(lr_images).latent_dist.sample()
                # reconstructed_images = vae.decode(latents).sample
                # loss = loss_fn(reconstructed_images, hr_images)

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        self._save_snapshot(vae, save_path)
        print(f"Fine-tuned VAE model saved at {save_path}")

        vae.eval()
        return vae

    def fine_tuning_Diffusion(self,
                            dataloader,
                            image_size,
                            epochs,
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

        # Set loss function and scheduler configuration
        loss_function = torch.nn.MSELoss()
        noise_steps = pipe.scheduler.config.num_train_timesteps

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
                if self.multiple_gpus:
                    latent_hr_image = pipe.vae.module.encode(hr_image).latent_dist.sample() * 0.18215
                else:
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

        self._save_snapshot(unet, save_path)
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

        return generated_image

class CombinedLoss(nn.Module):
    def __init__(self, perceptual_loss, mse_loss, alpha=0.5, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = perceptual_loss
        self.mse_loss = mse_loss
        self.alpha = alpha
        self.device = device

    def forward(self, input, target):
        perceptual_loss_value = self.perceptual_loss(input, target)
        mse_loss_value = self.mse_loss(input, target)
        combined_loss = self.alpha * perceptual_loss_value + (1 - self.alpha) * mse_loss_value
        return combined_loss
    
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
    

def launch(args):

    epochs = args.epochs
    batch_size = args.batch_size
    image_size = args.image_size
    learning_rate = args.lr
    dataset_path = args.dataset_path
    magnification_factor = args.magnification_factor
    Degradation_type = args.Degradation_type
    multiple_gpus = args.multiple_gpus
    Blur_radius = args.Blur_radius
    vae_snapshot_name = args.vae_snapshot_name
    diffusion_snapshot_name = args.diffusion_snapshot_name

    if vae_snapshot_name:
        if not vae_snapshot_name.endswith('.pt'):
            vae_snapshot_name += '.pt'
    if diffusion_snapshot_name:
        if not diffusion_snapshot_name.endswith('.pt'):
            diffusion_snapshot_name += '.pt'

            
    if Blur_radius.lower() != 'random':
        Blur_radius = float(Blur_radius)
        print('Using a blur radius of ', Blur_radius)
    else:
        print('Using random blur radius from a triangular distribution')

    print(f'Using {Degradation_type} degradation')
    
    if multiple_gpus:
        print('Using multiple GPUs')
        init_process_group(backend="nccl") # nccl stands for NVIDIA Collective Communication Library. It is used for distributed comunications across multiple GPUs.
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(int(device))
    else:   
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = 'mps'
        print('Using single GPU')
    
    if dataset_path:
        if Degradation_type.lower() == 'downblur':
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

            train_dataset = get_data_superres(train_path, magnification_factor, Blur_radius, False, 'PIL', transform)
            val_dataset = get_data_superres(valid_path, magnification_factor, Blur_radius, False, 'PIL', transform)
            
        elif Degradation_type.lower() == 'bsrgan':
            num_crops = 1

            train_path = f'{dataset_path}/train_original'
            valid_path = f'{dataset_path}/val_original'

            train_dataset = get_data_superres_BSRGAN(train_path, magnification_factor, image_size, num_crops=num_crops, degradation_type='BSR_plus', destination_folder=os.path.join(dataset_path+'_Dataset', 'train'))
            val_dataset = get_data_superres_BSRGAN(valid_path, magnification_factor, image_size, num_crops=num_crops, degradation_type='BSR_plus', destination_folder=os.path.join(dataset_path+'_Dataset', 'val'))

        elif Degradation_type.lower() == 'downblurnoise':
            train_path = f'{dataset_path}/train_original'
            valid_path = f'{dataset_path}/val_original'

            transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ])

            train_dataset = get_data_superres(train_path, magnification_factor, Blur_radius, True, 'PIL', transform)
            val_dataset = get_data_superres(valid_path, magnification_factor, Blur_radius, True, 'PIL', transform)
            # IF YOU WANT TO USE THE get_data BELOW, YOU NEED ALSO TO ADJUST THE STARTING TENSOR IN THE sample FUNCTION
            # train_dataset = get_data_superres_BSRGAN(train_path, magnification_factor, image_size, num_crops=num_crops, degradation_type='soft_BSR_plus', destination_folder=os.path.join(dataset_path+'_Dataset', 'train'))
            # val_dataset = get_data_superres_BSRGAN(valid_path, magnification_factor, image_size, num_crops=num_crops, degradation_type='soft_BSR_plus', destination_folder=os.path.join(dataset_path+'_Dataset', 'val'))
            
        else:
            raise ValueError('The degradation type must be either BSRGAN or DownBlur or DownBlurNoise')
        
        if multiple_gpus:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset),drop_last=True)
            # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False, sampler=DistributedSampler(val_dataset),drop_last=True)
        else:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    VAE_weight_path = os.path.join('models_run', vae_snapshot_name)
    Diffusion_weight_path = os.path.join('models_run', diffusion_snapshot_name)
    latent_diff_model = LatentDiffusion_superres(VAE_weight_path=VAE_weight_path,
                                                Diffusion_weight_path=Diffusion_weight_path,
                                                device = device,
                                                multiple_gpus=multiple_gpus)
    
    vae = latent_diff_model.fine_tuning_VAE(dataloader=train_loader,
                                    image_size=image_size,
                                    epochs=epochs,
                                    learning_rate=learning_rate)

    # unet = latent_diff_model.fine_tuning_Diffusion(dataloader=train_loader,
    #                                 image_size=image_size,
    #                                 epochs=epochs,
    #                                 learning_rate=learning_rate)

    if multiple_gpus:
        destroy_process_group()
    
    # lr_image = Image.open('celebA_10k/test_original/005044.jpg').resize((image_size//magnification_factor, image_size//magnification_factor))
    # lr_image = transforms.ToTensor()(lr_image).unsqueeze(0)
    # super_res_image = latent_diff_model.sample_superres(lr_image, num_inference_steps=100)
    # plt.imshow(super_res_image.squeeze().permute(1, 2, 0).cpu().numpy())
    # plt.savefig('super_res_image.png')

if __name__ == "__main__":
    import argparse  

    def str2bool(v):
        """Convert string to boolean."""
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--magnification_factor', type=int)
    parser.add_argument('--Degradation_type', type=str, default='DownBlur') # 'BSRGAN' or 'DownBlur' or 'DownBlurNoise'
    parser.add_argument('--multiple_gpus', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--Blur_radius', type=str, default='0.5')
    parser.add_argument('--vae_snapshot_name', type=str, default='VAE_finetuning')
    parser.add_argument('--diffusion_snapshot_name', type=str, default='Diffusion_finetuning')
    args = parser.parse_args()
    launch(args)


    

