import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import get_data_superres, get_data_superres_BSRGAN, get_data_superres_PLAIN, video_maker, CosineAnnealingWarmupRestarts
import copy

# from UNet_model_superres import Residual_Attention_UNet_superres, EMA
# from UNet_model_superres_VMHA import Residual_Attention_UNet_superres, Residual_VisionMultiheadAttention_UNet_superres, Residual_DiffiT_UNet_superres, EMA
from UNet_model_superres_CrossAttention import Residual_CrossAttention_UNet_superres, EMA
# from ViT_model import ViTModel

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from lpips import LPIPS  # Perceptual loss library

import warnings
import uuid
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import gc

from diffusers import StableDiffusionPipeline, UNet2DConditionModel

class Diffusion:
    def __init__(
            self,
            noise_schedule: str,
            model: nn.Module,
            vae_model: nn.Module,
            snapshot_path: str,
            VAE_weight_path: str,
            noise_steps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            device='cuda',
            magnification_factor=4,
            image_size=256,
            model_name='superres',
            Degradation_type='BSRGAN',
            multiple_gpus=False,
            ema_smoothing=False
            ):


        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.model_name = model_name
        self.magnification_factor = magnification_factor
        self.device = device
        self.multiple_gpus = multiple_gpus

        self.VAE_weight_path = VAE_weight_path
        self.snapshot_path = snapshot_path

        self.Degradation_type=Degradation_type
        
        self.ema_smoothing = ema_smoothing

        if model:
            self.model = model.to(self.device)

        if vae_model:
            self.vae_model = vae_model.to(self.device)

        # epoch_run is used by _save_snapshot and _load_snapshot to keep track of the current epoch
        self.epochs_run = 0

        # If a snapshot exists, we load it
        if snapshot_path:
            if os.path.exists(snapshot_path):
                print("Loading snapshot")
                self._load_snapshot()

        if VAE_weight_path:
            if os.path.exists(self.VAE_weight_path):
                print(f"Loading fine-tuned VAE model from {self.VAE_weight_path}...")
                self._load_snapshot_VAE(self.VAE_weight_path, self.vae_model)

        self.noise_schedule = noise_schedule

        if self.noise_schedule == 'linear':
            self.beta = self.prepare_noise_schedule().to(self.device) 
            self.alpha = 1. - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0) # Notice that beta is not just a number. It is a tensor of shape (noise_steps,).
        # If we are in the step t then we index the tensor with t. To get alpha_hat we compute the cumulative product of the tensor.

        elif self.noise_schedule == 'cosine':
            self.alpha_hat = self.prepare_noise_schedule().to(self.device)
            self.beta = self.from_alpha_hat_to_beta()
            self.alpha = 1. - self.beta

    def from_alpha_hat_to_beta(self):
        '''
        This function is necessary because it allows to get from the alpha hat that we got with the cosine schedule
        the alpha and the beta which are necessary in order to compute the denoised image during sampling.
        Check https://arxiv.org/pdf/2102.09672 at section 3.2 for more information.
        The reason we need this function is that with the linear schedule we start from beta, then we calculate alpha and so
        alpha hat, whereas with the cosine schedule we start from alpha hat, then we must calculate beta and so alpha
        because we need them to compute the denoised image.

        Input:
            alpha_hat: a tensor of shape (noise_steps,) that contains the alpha_hat values for each noise step.
        
        Output:
            beta: a tensor of shape (noise_steps,) that contains the beta values for each noise step.
        '''
        beta = []
        for t in range(len(self.alpha_hat)-1, 0, -1):
            beta.append(1 -(self.alpha_hat[t]/self.alpha_hat[t-1]))
        beta.append(1 - self.alpha_hat[0])
        beta =  torch.tensor(beta[::-1], dtype=self.alpha_hat.dtype, device=self.alpha_hat.device)
        return beta

    def prepare_noise_schedule(self):
        '''
        In this function we set the noise schedule to use. Basically, we need to know how much gaussian noise we want to add
        for each noise step.

        Input:
            noise_schedule: the name of the noise schedule to use. It can be either 'linear' or 'cosine'.

        Output:
            if noise_schedule == 'linear':
                self.beta: a tensor of shape (noise_steps,) that contains the beta values for each noise step.
            elif noise_schedule == 'cosine':
                self.alpha_hat: a tensor of shape (noise_steps,) that contains the alpha_hat values for each noise step.
        '''
        if self.noise_schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.noise_schedule == 'cosine':
            f_t = torch.cos(((((torch.arange(self.noise_steps)/self.noise_steps)+0.008)/(1+0.008))*torch.pi/2))**2 # Here we apply the formula of the OpenAI paper https://arxiv.org/pdf/2102.09672.pdf
            alpha_hat = f_t/f_t[0]  
            return alpha_hat

    def noise_images(self, x, t):
        '''
        ATTENTION: The error epsilon is random, but how much of it we add to move forward depends on the Beta schedule.

        Input:
            x: the image at time t=0
            t: the current timestep
        
        Output:
            x_t: the image at the current timestep (x_t)
            epsilon: the error that we add to x_t to move forward
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # Each None is a new dimension (e.g.
        # if a tensor has shape (2,3,4), a[None,None,:,None] will be shaped (1,1,2,1,3,4)). Basically, the dimensions are added where the None
        # are placed, and the : determines where the starting dimensions are placed (e.g. a[:,None,:,None] will be shaped (2,1,3,1,4),
        #a[None,None].shape=a[None,None,:].shape=a[None,None,:,:].shape=a[None,None,:,:,:].shape = (1,1,2,3,4)).
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x, dtype=torch.float32) # torch.randn_like() returns a tensor of the same shape of x with random values from a standard gaussian
        # (notice that the values inside x are not relevant)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        '''
        During the training we sample t from a Uniform discrete distribution (from 1 to T)

        For each image that I have in the training, I want to sample a timestep t from a uniform distribution
        (notice that it is not the same for each image). 

        Input:
            n: the number of images we want to sample the timesteps for (the batch size)

        Output:
            t: a tensor of shape (n,) that contains the timesteps for each image
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self,n, model, lr_img, generate_video=False):
        '''
        As the name suggests this function is used for sampling. Therefore we want to 
        loop backward. Moreover, notice that in the sample we want to perform EVERY STEP CONTIGUOUSLY
        while at training time we use the sample_timesteps() function to get just one random time step per batch.

        What we do is to predict the noise conditioned by the time step and by the low resolution image.

        Input:
            n: the number of images we want to sample
            lr_img: the low resolution image
            generate_video: if True, the function will produce a video with the generated NDVI images.
        
        Output:
            x: a tensor of shape (n, input_channels, self.image_size, self.image_size) with the generated images
        '''

        self.vae_model.eval()

        if len(lr_img.shape) < 4:
            lr_img = lr_img.unsqueeze(0)

        lr_img = F.interpolate(lr_img.to('cpu'), scale_factor=self.magnification_factor, mode='bicubic').to(self.device)
        
        with torch.no_grad():
            lr_img = self.vae_model.encode(lr_img).latent_dist.sample()

        model.eval() # disables dropout and batch normalization
        with torch.no_grad(): # disables gradient calculation
            if self.Degradation_type.lower() in {'downblur', 'bsrgan', 'downblurnoise'}:
                x = torch.randn((n, 4, self.image_size//8, self.image_size//8), device=self.device)
            else:
                raise ValueError('The degradation type must be either BSRGAN or DownBlur')

            x = x.to(self.device) 
            x = 0.05*lr_img+0.95*x

            frames = [] if generate_video else None  # Only allocate memory if needed
            
            shape_ = (n, 1, 1, 1)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): 
                t = torch.full((n,), i, dtype=torch.long, device=self.device) # tensor of shape (n) with all the elements equal to i.
                # Basically, each of the n image will be processed with the same integer time step t.

                predicted_noise = model(x, t, lr_img, self.magnification_factor).to(self.device)

                alpha = self.alpha[t].reshape(shape_)
                alpha_hat = self.alpha_hat[t].reshape(shape_)
                beta = self.beta[t].reshape(shape_)

                # If i>1 then we add noise to the image we have sampled (remember that from x_t we sample x_{t-1}).
                # If i==1 we sample x_0, which is the final image we want to generate, so we don't add noise.
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if generate_video:
                    frames.append(x.clone().detach().cpu())

        if generate_video:
            video_maker(frames, os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', 'video_denoising.mp4'), 100)
            del frames

        latent_sr_img = x
        latent_lr_img = lr_img

        # Delete unnecessary tensors to remove references
        del lr_img, x, frames, predicted_noise, noise  

        # Force Python garbage collection
        gc.collect()

        # Free unused GPU memory
        torch.cuda.empty_cache()

        # Perform inference without gradient tracking to save VRAM
        with torch.no_grad():
            sr_img = self.vae_model.decode(latent_sr_img).sample
            
        model.train() # enables dropout and batch normalization
        return latent_lr_img, latent_sr_img, sr_img

    def _save_snapshot(self, epoch, model):
        '''
        This function loads the model state and the current epoch from a snapshot.
        It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
        it from the last snapshot.

        Input:
            epoch: the current epoch
            model: the model to save

        Output:
            None
        '''
        if self.multiple_gpus:
            snapshot = {
                "MODEL_STATE": model.module.state_dict(),
                "EPOCHS_RUN": epoch,
                # "OPTIMIZER":self.optimizer.state_dict(),
                # "LR_SCHEDULER":self.lr_scheduler.state_dict(),
            }
        else:
            snapshot = {
                "MODEL_STATE": model.state_dict(),
                "EPOCHS_RUN": epoch,
                # "OPTIMIZER":self.optimizer.state_dict(),
                # "LR_SCHEDULER":self.lr_scheduler.state_dict(),
            }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _save_snapshot_VAE(self, model, snapshot_path):
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

    def _load_snapshot(self):
        '''
        This function loads the model state and the last epoch of training (so that we can restart the
        training at this point instead of restarting from 0) from a snapshot.
        It is a mandatory function in order to be fault tolerant. The reason is that if the training is interrupted, we can resume
        it from the last snapshot.
        '''
        if self.multiple_gpus:
            from collections import OrderedDict
            print(self.device)
            print(self.snapshot_path)

            snapshot = torch.load(self.snapshot_path, map_location='cpu', weights_only=True)
            model_state = OrderedDict((key.replace('module.', ''), value) for key, value in snapshot['MODEL_STATE'].items())
            self.model.module.load_state_dict(model_state)
            self.model.module.to(self.device)
        else:
            snapshot = torch.load(self.snapshot_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(snapshot["MODEL_STATE"])

        self.epochs_run = snapshot["EPOCHS_RUN"]
        # print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        print(f"Snapshot loaded from {self.snapshot_path}")

    def _load_snapshot_VAE(self, snapshot_path, model):
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

    def early_stopping(self, patience, epochs_without_improving):
        '''
        This function checks if the validation loss is increasing. If it is for more than patience times,
        then it returns True (that will correspond to breaking the training loop).
        '''
        if epochs_without_improving >= patience:
            print('Early stopping! Training stopped')
            return True

    def fine_tuning_VAE(self, dataloader, epochs, learning_rate):

        print("Fine-tuning VAE...")
        device = self.device
        vae = self.vae_model
        save_path = self.VAE_weight_path

        optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)

        loss_fn = vae_loss(device=device, lambda_rec=1.0, lambda_latent=0.5)

        vae.train()
        for epoch in range(epochs):
            pbar_dataloader = tqdm(dataloader, desc='Fine-tuning VAE', position=0)
            if self.multiple_gpus:
                pbar_dataloader.sampler.set_epoch(epoch) 
            total_loss = 0
            for i,(lr_images,hr_images) in enumerate(pbar_dataloader):
                lr_images = torch.stack([img for img in lr_images]).to(device).to(torch.float32)
                hr_images = torch.stack([img for img in hr_images]).to(device).to(torch.float32)

                if lr_images.device not in ['mps']:
                    lr_images = F.interpolate(lr_images, scale_factor=self.magnification_factor, mode='bicubic')
                else:
                    lr_images = F.interpolate(lr_images.to('cpu'), scale_factor=self.magnification_factor, mode='bicubic').to(self.device)
                
                if self.multiple_gpus:
                    latents_lr = self.vae_model.module.encode(lr_images).latent_dist.sample()
                    latents_hr = self.vae_model.module.encode(hr_images).latent_dist.sample()
                    reconstructed_lr = self.vae_model.module.decode(latents_lr).sample
                    reconstructed_hr = self.vae_model.module.decode(latents_hr).sample
                else:
                    latents_lr = self.vae_model.encode(lr_images).latent_dist.sample()
                    latents_hr = self.vae_model.encode(hr_images).latent_dist.sample()
                    reconstructed_lr = self.vae_model.decode(latents_lr).sample
                    reconstructed_hr = self.vae_model.decode(latents_hr).sample
                        
                loss = loss_fn(x_LR=lr_images, x_HR=hr_images, latents_lr=latents_lr, latents_hr=latents_hr, reconstructed_lr=reconstructed_lr, reconstructed_hr=reconstructed_hr)

                # gradient accumulation 
                if lr_images.shape[0] < 8: # If the batch size is smaller than 8 then use gradient accumulation
                    loss = loss/4

                loss.backward()
                total_loss += loss.item()

                # gradient accumulation
                if lr_images.shape[0] < 8: # If the batch size is smaller than 8 then use gradient accumulation
                    if (i+1) % 4 == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
            
            if self.multiple_gpus:
                if self.device == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

                    self._save_snapshot_VAE(vae, save_path)
                    print(f"Fine-tuned VAE model saved at {save_path}")
            else:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

                self._save_snapshot_VAE(vae, save_path)
                print(f"Fine-tuned VAE model saved at {save_path}")

        vae.eval()
        return vae

    def encoded_dataset_VAE(self, dataloader, save_path):
            import numpy as np
            self.vae_model.eval()
            os.makedirs(os.path.join(save_path, "lr_img"), exist_ok=True)
            os.makedirs(os.path.join(save_path, "hr_img"), exist_ok=True)
            pbar_dataloader = tqdm(dataloader, desc='Encoding dataset', position=0)
            for i,(lr_img,hr_img) in enumerate(pbar_dataloader):
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)
                lr_img = F.interpolate(lr_img.to('cpu'), scale_factor=self.magnification_factor, mode='bicubic').to(self.device)
                if self.multiple_gpus:
                    lr_img = self.vae_model.module.encode(lr_img).latent_dist.sample()
                    hr_img = self.vae_model.module.encode(hr_img).latent_dist.sample()
                else:
                    lr_img = self.vae_model.encode(lr_img).latent_dist.sample()
                    hr_img = self.vae_model.encode(hr_img).latent_dist.sample()
                for idx in range(lr_img.shape[0]):
                    unique_id = uuid.uuid4().hex
                    lr_img_to_save = lr_img[idx].permute(1,2,0).detach().cpu().numpy()
                    hr_img_to_save = hr_img[idx].permute(1,2,0).detach().cpu().numpy()
                    np.save(os.path.join(save_path, "lr_img",  f'{unique_id}'), lr_img_to_save)
                    np.save(os.path.join(save_path, "hr_img",  f'{unique_id}'), hr_img_to_save)
            
    def train(self, lr, epochs, check_preds_epoch, train_loader, val_loader, patience, loss, lr_scheduler=None):
        '''
        This function performs the training of the model, saves the snapshots and the model at the end of the training each self.every_n_epochs epochs.

        Input:
            lr: the learning rate
            epochs: the number of epochs
            check_preds_epoch: specifies the frequency, in terms of epochs, at which the model will perform predictions and save them. Moreover,
                if val_loader=None then the weights of the model will be saved at this frequency.
            train_loader: the training loader
            val_loader: the validation loader
            patience: the number of epochs after which the training will be stopped if the validation loss is increasing
            loss: the loss function to use
            lr_scheduler: the learning rate scheduler
        '''

        model = self.model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW is a variant of Adam that adds weight decay (L2 regularization)
        # Basically, weight decay is a regularization technique that penalizes large weights. It's a way to prevent overfitting. In AdamW, 
        # the weight decay is added to the gradient and not to the weights. This is because the weights are updated in a different way in AdamW.

        if self.ema_smoothing:
            ema = EMA(beta=0.995)
            ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        if loss == 'MSE':
            loss_function = nn.MSELoss()
        elif loss == 'MAE':
            loss_function = nn.L1Loss()
        elif loss == 'Huber':
            loss_function = nn.HuberLoss() 
        else:
            raise ValueError('The Loss must be either MSE or MAE or Huber')

        if lr_scheduler and lr_scheduler.lower() == 'cosine':
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=15,
                cycle_mult=2,
                max_lr=lr,
                min_lr=1e-5,
                warmup_steps=5,
                gamma=0.9
            )

        epochs_without_improving = 0
        best_loss = float('inf')  

        for epoch in range(self.epochs_run, epochs):
            if self.multiple_gpus:
                train_loader.sampler.set_epoch(epoch) # ensures that the data is shuffled in a consistent manner across multiple epochs (it is useful just for the DistributedSampler)

            b_sz = len(next(iter(train_loader))[0])
            print(f"\n\n[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}")
            
            pbar_train = tqdm(train_loader,desc='Training', position=0)
            if val_loader is not None:
                pbar_val = tqdm(val_loader,desc='Validation', position=0)

            running_train_loss = 0.0
            running_val_loss = 0.0

            model.train()
            for i,(lr_img,hr_img) in enumerate(pbar_train):
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

                t = self.sample_timesteps(hr_img.shape[0]).to(self.device)
                # t is a unidimensional tensor of shape (hr_img.shape[0] that is the batch_size) with random integers from 1 to noise_steps.
                x_t, noise = self.noise_images(hr_img, t) # get the noisy images

                optimizer.zero_grad() # set the gradients to 0
                predicted_noise = model(x_t, t, lr_img, self.magnification_factor) 

                train_loss = loss_function(predicted_noise, noise)
                train_loss.backward() # compute the gradients
                optimizer.step() # update the weights

                if self.ema_smoothing:
                    ema.step_ema(ema_model, model)
                
                pbar_train.set_postfix(LOSS=train_loss.item()) # set_postfix just adds a message or value displayed after the progress bar. In this case the loss of the current batch.
            
                running_train_loss += train_loss.item()

            if lr_scheduler and lr_scheduler.lower() != 'none':
                scheduler.step()

            running_train_loss /= len(train_loader) # at the end of each epoch I want the average loss
            print(f"Epoch {epoch}: Running Train ({loss}) {running_train_loss}; LR: {optimizer.param_groups[0]['lr']}")

            # IF THERE ARE MULTIPLE GPUs, MAKE JUST THE FIRST ONE SAVE THE SNAPSHOT AND COMPUTE THE PREDICTIONS TO AVOID REDUNDANCY
            # IN THE ELSE STATEMENT, THERE IS EXACTLY THE SAME. 
            if self.multiple_gpus:
                if self.device==0 and epoch % check_preds_epoch == 0:
                    if val_loader is None: # if there is no validation loader, then we save the weights at the frequency check_preds_epoch
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)
                        else:
                            self._save_snapshot(epoch, model)
            else:
                if epoch % check_preds_epoch == 0:
                    if val_loader is None: # if there is no validation loader, then we save the weights at the frequency check_preds_epoch
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)
                        else:
                            self._save_snapshot(epoch, model)

            if val_loader is not None:
                with torch.no_grad():
                    model.eval()
                    
                    for (lr_img,hr_img) in pbar_val:
                        lr_img = lr_img.to(self.device)
                        hr_img = hr_img.to(self.device)

                        t = self.sample_timesteps(hr_img.shape[0]).to(self.device) # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size)with random integers from 1 to noise_steps.
                        x_t, noise = self.noise_images(hr_img, t) # get batch_size noise images
                        
                        if self.ema_smoothing:
                            predicted_noise = ema_model(x_t, t, lr_img, self.magnification_factor)
                        else:
                            predicted_noise = model(x_t, t, lr_img, self.magnification_factor) 
                        
                        val_loss = loss_function(predicted_noise, noise)

                        pbar_val.set_postfix(LOSS=val_loss.item()) # set_postfix just adds a message or value
                        # displayed after the progress bar. In this case the loss of the current batch.

                        running_val_loss += val_loss.item()

                    running_val_loss /= len(val_loader)
                    print(f"Epoch {epoch}: Running Val loss ({loss}){running_val_loss}")

                if running_val_loss < best_loss - 0:
                    best_loss = running_val_loss
                    epochs_without_improving = 0
                    if self.multiple_gpus:
                        if self.device==0:
                            if self.ema_smoothing:
                                self._save_snapshot(epoch, ema_model)
                            else:
                                self._save_snapshot(epoch, model)
                    else:
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)
                        else:
                            self._save_snapshot(epoch, model)  
                else:
                    epochs_without_improving += 1

                if self.early_stopping(patience, epochs_without_improving):
                    break
            print('Epochs without improving: ', epochs_without_improving)
    
    def fine_tuning_UNet(self, model, lr, epochs, check_preds_epoch, train_loader, val_loader, patience, loss, lr_scheduler=None):
        '''
        This function performs the training of the model, saves the snapshots and the model at the end of the training each self.every_n_epochs epochs.

        Input:
            lr: the learning rate
            epochs: the number of epochs
            check_preds_epoch: specifies the frequency, in terms of epochs, at which the model will perform predictions and save them. Moreover,
                if val_loader=None then the weights of the model will be saved at this frequency.
            train_loader: the training loader
            val_loader: the validation loader
            patience: the number of epochs after which the training will be stopped if the validation loss is increasing
            loss: the loss function to use
            lr_scheduler: the learning rate scheduler
        '''

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW is a variant of Adam that adds weight decay (L2 regularization)
        # Basically, weight decay is a regularization technique that penalizes large weights. It's a way to prevent overfitting. In AdamW, 
        # the weight decay is added to the gradient and not to the weights. This is because the weights are updated in a different way in AdamW.

        if self.ema_smoothing:
            ema = EMA(beta=0.995)
            ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        if loss == 'MSE':
            loss_function = nn.MSELoss()
        elif loss == 'MAE':
            loss_function = nn.L1Loss()
        elif loss == 'Huber':
            loss_function = nn.HuberLoss() 
        else:
            raise ValueError('The Loss must be either MSE or MAE or Huber')

        if lr_scheduler and lr_scheduler.lower() == 'cosine':
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=15,
                cycle_mult=2,
                max_lr=lr,
                min_lr=1e-5,
                warmup_steps=5,
                gamma=0.9
            )

        epochs_without_improving = 0
        best_loss = float('inf')  

        for epoch in range(self.epochs_run, epochs):
            if self.multiple_gpus:
                train_loader.sampler.set_epoch(epoch) # ensures that the data is shuffled in a consistent manner across multiple epochs (it is useful just for the DistributedSampler)

            b_sz = len(next(iter(train_loader))[0])
            print(f"\n\n[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}")
            
            pbar_train = tqdm(train_loader,desc='Training', position=0)
            if val_loader is not None:
                pbar_val = tqdm(val_loader,desc='Validation', position=0)

            running_train_loss = 0.0
            running_val_loss = 0.0

            model.train()
            for i,(lr_img,hr_img) in enumerate(pbar_train):
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)
                # lr_img = F.interpolate(lr_img, size=(64, 64), mode="bilinear", align_corners=False)
                # hr_img = F.interpolate(hr_img, size=(64, 64), mode="bilinear", align_corners=False)
                # lr_img = lr_img.half() 
                # hr_img = hr_img.half()

                t = self.sample_timesteps(hr_img.shape[0]).to(self.device)
                # t is a unidimensional tensor of shape (hr_img.shape[0] that is the batch_size) with random integers from 1 to noise_steps.
                x_t, noise = self.noise_images(hr_img, t) # get the noisy images

                optimizer.zero_grad() # set the gradients to 0
                predicted_noise = model(x_t, t, encoder_hidden_states=lr_img).sample

                train_loss = loss_function(predicted_noise, noise)
                
                train_loss.backward() # compute the gradients
                optimizer.step() # update the weights

                if self.ema_smoothing:
                    ema.step_ema(ema_model, model)
                
                pbar_train.set_postfix(LOSS=train_loss.item()) # set_postfix just adds a message or value displayed after the progress bar. In this case the loss of the current batch.
            
                running_train_loss += train_loss.item()

            if lr_scheduler and lr_scheduler.lower() != 'none':
                scheduler.step()

            running_train_loss /= len(train_loader) # at the end of each epoch I want the average loss
            print(f"Epoch {epoch}: Running Train ({loss}) {running_train_loss}; LR: {optimizer.param_groups[0]['lr']}")

            # IF THERE ARE MULTIPLE GPUs, MAKE JUST THE FIRST ONE SAVE THE SNAPSHOT AND COMPUTE THE PREDICTIONS TO AVOID REDUNDANCY
            # IN THE ELSE STATEMENT, THERE IS EXACTLY THE SAME. 
            if self.multiple_gpus:
                if self.device==0 and epoch % check_preds_epoch == 0:
                    if val_loader is None: # if there is no validation loader, then we save the weights at the frequency check_preds_epoch
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)
                        else:
                            self._save_snapshot(epoch, model)
            else:
                if epoch % check_preds_epoch == 0:
                    if val_loader is None: # if there is no validation loader, then we save the weights at the frequency check_preds_epoch
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)
                        else:
                            self._save_snapshot(epoch, model)

            if val_loader is not None:
                with torch.no_grad():
                    model.eval()
                    
                    for (lr_img,hr_img) in pbar_val:
                        lr_img = lr_img.to(self.device)
                        hr_img = hr_img.to(self.device)

                        t = self.sample_timesteps(hr_img.shape[0]).to(self.device) # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size)with random integers from 1 to noise_steps.
                        x_t, noise = self.noise_images(hr_img, t) # get batch_size noise images
                        
                        if self.ema_smoothing:
                            predicted_noise = ema_model(x_t, t, lr_img).sample
                            
                        else:
                            predicted_noise = model(x_t, t, lr_img).sample 
                        
                        val_loss = loss_function(predicted_noise, noise)

                        pbar_val.set_postfix(LOSS=val_loss.item()) # set_postfix just adds a message or value
                        # displayed after the progress bar. In this case the loss of the current batch.

                        running_val_loss += val_loss.item()

                    running_val_loss /= len(val_loader)
                    print(f"Epoch {epoch}: Running Val loss ({loss}){running_val_loss}")

                if running_val_loss < best_loss - 0:
                    best_loss = running_val_loss
                    epochs_without_improving = 0
                    if self.multiple_gpus:
                        if self.device==0:
                            if self.ema_smoothing:
                                self._save_snapshot(epoch, ema_model)
                            else:
                                self._save_snapshot(epoch, model)
                    else:
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)
                        else:
                            self._save_snapshot(epoch, model)  
                else:
                    epochs_without_improving += 1

                if self.early_stopping(patience, epochs_without_improving):
                    break
            print('Epochs without improving: ', epochs_without_improving)


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
        images = (images - torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        return images

class vae_loss(nn.Module):
    def __init__(self, device, lambda_rec=1.0, lambda_latent=0.5):
        super(vae_loss, self).__init__()
        self.lpips_loss = LPIPS(net='vgg').to(device)
        self.lambda_rec = lambda_rec
        self.lambda_latent = lambda_latent
        
    def forward(self, x_LR, x_HR, latents_lr, latents_hr, reconstructed_lr, reconstructed_hr):
        rec_loss = self._reconstruction_loss(x_LR, x_HR, reconstructed_lr, reconstructed_hr)
        latent_loss = self._latent_consistency_loss(latents_lr, latents_hr)
        total_loss = self.lambda_rec * rec_loss + self.lambda_latent * latent_loss
        return total_loss

    def _reconstruction_loss(self, x_LR, x_HR, reconstructed_lr, reconstructed_hr):
        rec_loss = (F.l1_loss(reconstructed_hr, x_HR) + F.l1_loss(reconstructed_lr, x_LR) +
            self.lpips_loss(reconstructed_hr, x_HR).mean())
        return rec_loss
    
    def _latent_consistency_loss(self, latents_lr, latents_hr):
        '''
        The reason for this loss is that we want the latent space of the low resolution images
        to be similar to the latent space of the high resolution images. This is because the semantic
        information should be the same in both the low and high resolution images and in the latent 
        space we want all this information to be located and all the perceptual information to be
        discarded.
        '''
        latent_loss = F.mse_loss(latents_lr, latents_hr)
        return latent_loss 

def dataloader_PRE_encoding_maker(dataset_path, Degradation_type, image_size, magnification_factor, Blur_radius, num_crops=1, batch_size=16, multiple_gpus=False):
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
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False, sampler=DistributedSampler(val_dataset),drop_last=True)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return train_loader, val_loader

def dataloader_POST_encoding_maker(dataset_path, batch_size, multiple_gpus):
    dataset = get_data_superres_PLAIN(dataset_path)
    if multiple_gpus:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset),drop_last=True)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def UNet_model_maker(UNet_type, input_channels, output_channels, device, image_size):
    if UNet_type.lower() == 'residual attention unet':
        print('Using Residual Attention UNet')
        model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
    elif UNet_type.lower() == 'residual cross attention unet':
        print('Using Residual Cross Attention UNet')
        model = Residual_CrossAttention_UNet_superres(input_channels, output_channels, device).to(device)
    elif UNet_type.lower() == 'residual multihead attention unet':
        print('Using Residual MultiHead Attention UNet')
        # model = Residual_MultiHeadAttention_UNet_superres(input_channels, output_channels, device).to(device)
    elif UNet_type.lower() == 'residual vision multihead attention unet':
        print('Using Residual Vision MultiHead Attention UNet')
        model = Residual_VisionMultiheadAttention_UNet_superres(input_channels, output_channels, image_size=image_size,device=device).to(device) # The images must be squared
    elif UNet_type.lower() == 'vision transformer':
        print('Using Vision Transformer (noUnet)')
        model = ViTModel(image_channels=input_channels, device=device).to(device)
    elif UNet_type.lower() == 'diffit unet':
        print('Using Diffit UNet')
        model = Residual_DiffiT_UNet_superres(input_channels, output_channels, device).to(device)
    else:
        raise ValueError('The UNet type must be Residual Attention UNet or Residual MultiHead Attention UNet or Residual Vision MultiHeadAttention UNet superres or Diffit UNet')
    
    return model

def super_resolution_sampling(diffusion_class, UNet_model, lr_img, generate_video=False, hr_img=None, save_path=None):
    if hr_img:
        fig, axs = plt.subplots(2,3, figsize=(15,15))
    else:
        fig, axs = plt.subplots(1,4, figsize=(15,15))
    axs = axs.ravel()

    latent_lr_img, latent_sr_img, superres_img = diffusion_class.sample(n=1,model=UNet_model, lr_img=lr_img, input_channels=lr_img.shape[0], generate_video=generate_video)

    axs[0].imshow(lr_img.permute(1,2,0).detach().cpu().numpy())
    axs[0].set_title('Low resolution image')
    axs[1].imshow(latent_lr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
    axs[1].set_title('Low resolution latent')
    axs[2].imshow(superres_img[0].permute(1,2,0).detach().cpu().numpy())
    axs[2].set_title('Super resolution image')
    axs[3].imshow(latent_sr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
    axs[3].set_title('Super resolution latent')
    if hr_img:
        axs[4].imshow(hr_img.permute(1,2,0).detach().cpu().numpy())
        axs[4].set_title('High resolution image')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return superres_img

def VAE_model_maker(device):
    vae_model_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(vae_model_path)
    vae_model = pipe.vae.to(device)
    return vae_model

def VAE_finetuning(dataset_path, Degradation_type, image_size, magnification_factor, Blur_radius, VAE_weight_path, num_crops=1, batch_size=16, multiple_gpus=False, device='cuda'):

    train_loader, val_loader = dataloader_PRE_encoding_maker(dataset_path=dataset_path, Degradation_type=Degradation_type,
                                                 image_size=image_size, magnification_factor=magnification_factor,
                                                   Blur_radius=Blur_radius, num_crops=num_crops, batch_size=batch_size, 
                                                     multiple_gpus=multiple_gpus)
    vae_model = VAE_model_maker(device)
        
    if multiple_gpus:
        vae_model = DDP(vae_model, device_ids=[device], find_unused_parameters=True) 

    diffusion = Diffusion(
        noise_schedule=None, model=None, vae_model=vae_model,
        snapshot_path=None,
        VAE_weight_path=VAE_weight_path,
        noise_steps=None, beta_start=None, beta_end=None, 
        magnification_factor=magnification_factor,device=device,
        image_size=image_size, model_name=None, Degradation_type=Degradation_type,
        multiple_gpus=multiple_gpus, ema_smoothing=None)
        
    diffusion.fine_tuning_VAE(train_loader, epochs=40, learning_rate=1e-4)

    ########## ENCODE DATASET AND SAVE IT ##########
    encoded_images_train_save_path = os.path.join(dataset_path+'_VAE_encoded', "train_original")
    encoded_images_val_save_path = os.path.join(dataset_path+'_VAE_encoded', "val_original")
    if os.path.exists(os.path.join(encoded_images_train_save_path, 'lr_img')):
        if len(os.listdir(os.path.join(encoded_images_train_save_path, 'lr_img'))) == 0:
            diffusion.encoded_dataset_VAE(dataloader=train_loader, save_path=encoded_images_train_save_path)
            diffusion.encoded_dataset_VAE(dataloader=val_loader, save_path=encoded_images_val_save_path)
    else:
        diffusion.encoded_dataset_VAE(dataloader=train_loader, save_path=encoded_images_train_save_path)
        diffusion.encoded_dataset_VAE(dataloader=val_loader, save_path=encoded_images_val_save_path)

    ########## RENAME IMAGES (OPTIONAL) ##########
    # for set_path in [encoded_images_train_save_path, encoded_images_val_save_path]:
    #     for i, img_name in tqdm(enumerate(os.listdir(os.path.join(set_path, "lr_img"))), desc='Renaming images', position=0):
    #         if img_name.endswith('.npy'):
    #                 os.rename(os.path.join(set_path, "lr_img", img_name), os.path.join(set_path, "lr_img", str(i+50000)+".npy"))
    #                 os.rename(os.path.join(set_path, "hr_img", img_name), os.path.join(set_path, "hr_img", str(i+50000)+".npy"))

def Diffusion_training(snapshot_folder_path, model_name, snapshot_name,
                        noise_steps, ema_smoothing, magnification_factor,  
                            UNet_type, input_channels, output_channels, 
                                batch_size, image_size, multiple_gpus, 
                                    noise_schedule, dataset_path, lr,
                                     epochs,check_preds_epoch, patience,
                                      loss, lr_scheduler, device):

    os.makedirs(snapshot_folder_path, exist_ok=True)
    os.makedirs(os.path.join(os.curdir, 'models_run', model_name, 'results'), exist_ok=True)

    model = UNet_model_maker(UNet_type, input_channels, output_channels, device, image_size)
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    if multiple_gpus:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

    diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model, vae_model=None,
        snapshot_path=snapshot_path,
        VAE_weight_path=None,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
        magnification_factor=magnification_factor,device=device,
        image_size=image_size, model_name=model_name, Degradation_type=None,
        multiple_gpus=multiple_gpus, ema_smoothing=ema_smoothing)
    
    assert "_encoded" in dataset_path, "The dataset path must contain '_encoded' in the name"
    encoded_images_train_save_path = os.path.join(dataset_path, "train_original")
    encoded_images_val_save_path = os.path.join(dataset_path, "val_original")

    ########## CREATE DATALOADERS FOR THE POST-ENCODING MODEL ##########
    train_loader = dataloader_POST_encoding_maker(encoded_images_train_save_path, batch_size, multiple_gpus)
    # val_loader = dataloader_POST_encoding_maker(encoded_images_val_save_path, batch_size, multiple_gpus)
    val_loader = None
    ########## TRAIN DIFFUSION MODEL ##########
    diffusion.train(
        lr=lr, epochs=epochs, check_preds_epoch=check_preds_epoch,
        train_loader=train_loader, val_loader=val_loader, patience=patience, loss=loss,
        lr_scheduler=lr_scheduler)
    
    if multiple_gpus:
        destroy_process_group()

def sampling_test(snapshot_folder_path, model_name, snapshot_name, UNet_type,
              input_channels, output_channels, image_size, 
              noise_schedule, noise_steps, magnification_factor,
              Degradation_type, dataset_path, Blur_radius, VAE_weight_path,
              num_crops=None, generate_video=False,
             device='cuda'):
    
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

    train_loader, val_loader = dataloader_PRE_encoding_maker(dataset_path=dataset_path, Degradation_type=Degradation_type,
                                                 image_size=image_size, magnification_factor=magnification_factor,
                                                   Blur_radius=Blur_radius, num_crops=num_crops, batch_size=1, 
                                                    multiple_gpus=False)
    ######### SAMPLING ##########
    fig, axs = plt.subplots(5,5, figsize=(15,15))
    for i in range(5):
        lr_img = train_loader.dataset[i][0]
        hr_img = train_loader.dataset[i][1]

        latent_lr_img, latent_sr_img, superres_img = diffusion.sample(n=1,model=model, lr_img=lr_img, generate_video=generate_video)

        axs[i,0].imshow(lr_img.permute(1,2,0).detach().cpu().numpy())
        axs[i,0].set_title('Low resolution image')
        axs[i,1].imshow(latent_lr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
        axs[i,1].set_title('Low resolution latent')
        axs[i,2].imshow(hr_img.permute(1,2,0).detach().cpu().numpy())
        axs[i,2].set_title('High resolution image')
        axs[i,3].imshow(superres_img[0].permute(1,2,0).detach().cpu().numpy())
        axs[i,3].set_title('Super resolution image')
        axs[i,4].imshow(latent_sr_img[0][:3,:,:].permute(1,2,0).detach().cpu().numpy())
        axs[i,4].set_title('Super resolution latent')

    plt.savefig(os.path.join(os.getcwd(), 'models_run', model_name, 'results', 'superres_results.png'))

def launch(args):
    '''
    This function is the main and call the training, the sampling and all the other functions in the Diffusion class.

    Input:
        image_size: the size of the high resolution images
        dataset_path: the path of the dataset
        batch_size: the batch size
        lr: the learning rate
        lr_scheduler: the learning rate scheduler
        epochs: the number of epochs
        noise_schedule: the noise schedule (linear, cosine)
        check_preds_epoch: specifies the frequency, in terms of epochs, at which the model will perform predictions and save them. Moreover,
            if val_loader=None then the weights of the model will be saved at this frequency.
        snapshot_name: the name of the snapshot file
        snapshot_folder_path: the folder path where the snapshots will be saved
        model_name: the name of the model
        noise_steps: the number of noise steps
        patience: the number of epochs after which the training will be stopped if the validation loss is increasing
        input_channels: the number of input channels
        output_channels: the number of output channels
        generate_video: if True, the function will produce a video with the generated NDVI images.
        magnification_factor: the magnification factor (i.e. the factor by which the image is magnified in the super-resolution task)
        loss: the loss function to use
        UNet_type: the type of UNet to use (attention unet, residual attention unet, residual attention unet 2, residual multihead attention unet, residual vision multihead attention unet)
        Degradation_type: the type of degradation to use (downblur, bsrgan, downblurnoise)
        num_crops: the number of crops to use
        multiple_gpus: if True, the function will use multiple GPUs
        ema_smoothing: if True, the function will use EMA smoothing

    Output:
        None
    '''
    image_size = args.image_size
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    epochs = args.epochs
    noise_schedule = args.noise_schedule
    check_preds_epoch = args.check_preds_epoch
    snapshot_name = args.snapshot_name
    snapshot_folder_path = args.snapshot_folder_path
    model_name = args.model_name
    noise_steps = args.noise_steps
    patience = args.patience
    input_channels, output_channels = args.inp_out_channels, args.inp_out_channels
    generate_video = args.generate_video
    magnification_factor = args.magnification_factor
    loss = args.loss
    UNet_type = args.UNet_type
    Degradation_type = args.Degradation_type
    num_crops = args.num_crops
    multiple_gpus = args.multiple_gpus
    ema_smoothing = args.ema_smoothing
    Blur_radius = args.Blur_radius
    VAE_weight_path = args.VAE_weight_path

    if Blur_radius:
        if Blur_radius.lower() != 'random':
            Blur_radius = float(Blur_radius)
            print('Using a blur radius of ', Blur_radius)
        else:
            Blur_radius="random"
            print('Using random blur radius from a triangular distribution')

    print(f'Using {Degradation_type} degradation')
    
    if lr_scheduler and lr_scheduler.lower() != 'none':
        print(f'Using {lr_scheduler} learning rate scheduler')

    if ema_smoothing:
        print(f'Using EMA smoothing')
    else:
        print(f'Not using EMA smoothing')

    if multiple_gpus:
        print('Using multiple GPUs')
        init_process_group(backend="nccl") # nccl stands for NVIDIA Collective Communication Library. It is used for distributed comunications across multiple GPUs.
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(int(device))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f'Using single device: {device}')

    # VAE_finetuning(dataset_path=dataset_path, Degradation_type=Degradation_type, image_size=image_size,
    #                 magnification_factor=magnification_factor, Blur_radius=Blur_radius,VAE_weight_path=VAE_weight_path, num_crops=num_crops,
    #                     batch_size=batch_size, multiple_gpus=multiple_gpus, device=device)
    
    # Diffusion_training(snapshot_folder_path=snapshot_folder_path, model_name=model_name, snapshot_name=snapshot_name,
    #                     noise_steps=noise_steps, ema_smoothing=ema_smoothing, magnification_factor=magnification_factor,  
    #                         UNet_type=UNet_type, input_channels=input_channels, output_channels=output_channels, 
    #                             batch_size=batch_size, image_size=image_size, multiple_gpus=multiple_gpus, 
    #                                 noise_schedule=noise_schedule, dataset_path=dataset_path, lr=lr,
    #                                  epochs=epochs,check_preds_epoch=check_preds_epoch, patience=patience,
    #                                   loss=loss, lr_scheduler=lr_scheduler, device=device)
    
    sampling_test(snapshot_folder_path=snapshot_folder_path, model_name=model_name, snapshot_name=snapshot_name, UNet_type=UNet_type,
                    input_channels=input_channels, output_channels=output_channels, image_size=image_size, 
                        noise_schedule=noise_schedule, noise_steps=noise_steps, magnification_factor=magnification_factor,
                            Degradation_type=Degradation_type, dataset_path=dataset_path, Blur_radius=Blur_radius, VAE_weight_path=VAE_weight_path, generate_video=generate_video, device=device)


if __name__ == '__main__':
    import argparse  

    def str2bool(v):
        """Convert string to boolean."""
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_scheduler', type=str, default=None)
    parser.add_argument('--check_preds_epoch', type=int, default=None)
    parser.add_argument('--noise_schedule', type=str, default=None)
    parser.add_argument('--snapshot_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--noise_steps', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--inp_out_channels', type=int, default=None) # input channels must be the same of the output channels
    parser.add_argument('--generate_video', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--magnification_factor', type=int, default=None)
    parser.add_argument('--UNet_type', type=str, default=None) # 'Residual Attention UNet' or 'Residual MultiHead Attention UNet' or 'Residual Vision MultiHead Attention UNet'
    parser.add_argument('--Degradation_type', type=str, default=None) # 'BSRGAN' or 'DownBlur' or 'DownBlurNoise'
    parser.add_argument('--num_crops', type=int, default=None)
    parser.add_argument('--multiple_gpus', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--ema_smoothing', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--Blur_radius', type=str, default=None)
    parser.add_argument('--VAE_weight_path', type=str, default=None)
    args = parser.parse_args()
    if args.model_name:
        args.snapshot_folder_path = os.path.join(os.curdir, 'models_run', args.model_name, 'weights')
    else:
        args.snapshot_folder_path = None

    if args.VAE_weight_path:
        args.VAE_weight_path = os.path.join('models_run', args.VAE_weight_path)
    else:
        args.VAE_weight_path = None

    launch(args)