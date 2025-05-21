import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, models
from utils import get_data_SAR_TO_NDVI, video_maker, CosineAnnealingWarmupRestarts, compute_global_min_max, GlobalMinMaxScaler
import copy
import numpy as np
from UNet_model_SAR_TO_NDVI_CrossAttention import Residual_CrossAttention_UNet_SAR_TO_NDVI, EMA

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
import clip
from torchvision.transforms.functional import resize

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
            image_size=224,
            model_name='SAR_TO_NDVI',
            multiple_gpus=False,
            ema_smoothing=False,
            ):
    
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.model_name = model_name
        self.device = device
        self.multiple_gpus = multiple_gpus
        
        self.VAE_weight_path = VAE_weight_path
        self.snapshot_path = snapshot_path

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

    def xt_to_x0(self, x_t, t, noise_pred):
        '''
        This function is used to compute the x_0 from the x_t and the predicted noise. 
        It is used in the training phase to compute the CLIP loss.

        Input:
            x_t: the image at time t
            t: the current timestep
            noise_pred: the predicted noise

        Output:
            x_0: the image at time t=0
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return (x_t - sqrt_one_minus_alpha_hat * noise_pred) / sqrt_alpha_hat
    
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
    
    def sample(self, n, model, SAR_img, generate_video=False):
        '''
        As the name suggests this function is used for sampling. Therefore we want to 
        loop backward. Moreover, notice that in the sample we want to perform EVERY STEP CONTIGUOUSLY,
        while at training time we use the sample_timesteps() function to get just one random time step per batch.

        What we do is to predict the noise conditioned by the time step and by the SAR image.

        Input:
            n: the number of images we want to sample
            SAR_img: the SAR_img (shaped (SAR_channels, self.image_size, self.image_size)) 
            generate_video: if True, the function will produce a video with the generated NDVI images.
        
        Output:
            x: a tensor of shape (n, NDVI_channels, self.image_size, self.image_size) with the generated images
        '''
        self.vae_model.eval()
        SAR_img = SAR_img.to(self.device).unsqueeze(0)
        SAR_img = self.vae_model.encode(SAR_img)
        model.eval() # disables dropout and batch normalization
        with torch.no_grad(): # disables gradient calculation
            x = torch.randn((n, 4, self.image_size//8, self.image_size//8), device=self.device)

            frames = [] if generate_video else None  # Only allocate memory if needed

            shape_ = (n, 1, 1, 1)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): 
                t = torch.full((n,), i, dtype=torch.long, device=self.device) # tensor of shape (n) with all the elements equal to i.
                # Basically, each of the n image will be processed with the same integer time step t.

                predicted_noise = model(x, t, SAR_img)

                alpha = self.alpha[t].reshape(shape_)
                alpha_hat = self.alpha_hat[t].reshape(shape_)
                beta = self.beta[t].reshape(shape_)

                # If i>1 then we add noise to the image we have sampled (remember that from x_t we sample x_{t-1}).
                # If i==1 we sample x_0, which is the final image we want to generate, so we don't add noise.
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if generate_video == True:
                    frames.append(x)
                # if i % 100 == 0:
                #     import ipdb; ipdb.set_trace()
                #     fig, axs = plt.subplots(1,2);axs[0].imshow(SAR_img[0][:3,:,:].permute(1,2,0).detach().cpu());axs[1].imshow(x[0][:3,:,:].permute(1,2,0).detach().cpu())
        if generate_video == True:
            video_maker(frames, os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', 'video_denoising.mp4'), 100)
            del frames

        global_min = float(np.load(os.path.join(os.path.dirname(os.path.dirname(self.snapshot_path)), "global_min.npy")))
        global_max = float(np.load(os.path.join(os.path.dirname(os.path.dirname(self.snapshot_path)), "global_max.npy")))
        
        x = (x+1)*(global_max-global_min)/2 +global_min

        latent_NDVI_img = x
        latent_SAR_img = SAR_img

        # Delete unnecessary tensors to remove references
        del SAR_img, x, predicted_noise, noise  

        # Force Python garbage collection
        gc.collect()

        # Free unused GPU memory
        torch.cuda.empty_cache()

        # Perform inference without gradient tracking to save VRAM
        with torch.no_grad():
            NDVI_pred_img = self.vae_model.decode(latent_NDVI_img)
        
        model.train() # enables dropout and batch normalization
        return NDVI_pred_img

    def _save_snapshot(self, epoch, model):
        '''
        This function saves the model state and the current epoch.
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

        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

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
            for i,(SAR_img,NDVI_img) in enumerate(pbar_dataloader):
                SAR_img = torch.stack([img for img in SAR_img]).to(device).to(torch.float32)
                NDVI_img = torch.stack([img for img in NDVI_img]).to(device).to(torch.float32)
                
                if self.multiple_gpus:
                    latents_SAR = self.vae_model.module.encode(SAR_img)
                    latents_NDVI = self.vae_model.module.encode(NDVI_img)
                    reconstructed_SAR = self.vae_model.module.decode(latents_SAR)
                    reconstructed_NDVI = self.vae_model.module.decode(latents_NDVI)
                else:
                    latents_SAR = self.vae_model.encode(SAR_img)
                    latents_NDVI = self.vae_model.encode(NDVI_img)
                    reconstructed_SAR = self.vae_model.decode(latents_SAR)
                    reconstructed_NDVI = self.vae_model.decode(latents_NDVI)
                        
                loss = loss_fn(x_SAR=SAR_img, x_NDVI=NDVI_img, latents_SAR=latents_SAR, latents_NDVI=latents_NDVI, reconstructed_SAR=reconstructed_SAR, reconstructed_NDVI=reconstructed_NDVI)
                # gradient accumulation 
                if SAR_img.shape[0] < 8: # If the batch size is smaller than 8 then use gradient accumulation
                    loss = loss/4

                loss.backward()
                total_loss += loss.item()

                # gradient accumulation
                if SAR_img.shape[0] < 8: # If the batch size is smaller than 8 then use gradient accumulation
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
            os.makedirs(os.path.join(save_path, "sar"), exist_ok=True)
            os.makedirs(os.path.join(save_path, "opt"), exist_ok=True)
            pbar_dataloader = tqdm(dataloader, desc='Encoding dataset', position=0)
            for i,(SAR_img,NDVI_img) in enumerate(pbar_dataloader):
                SAR_img = SAR_img.to(self.device)
                NDVI_img = NDVI_img.to(self.device)
                if self.multiple_gpus:
                    SAR_img = self.vae_model.module.encode(SAR_img)
                    NDVI_img = self.vae_model.module.encode(NDVI_img)
                else:
                    SAR_img = self.vae_model.encode(SAR_img) 
                    NDVI_img = self.vae_model.encode(NDVI_img)

                for idx in range(SAR_img.shape[0]):
                    unique_id = uuid.uuid4().hex
                    SAR_img_to_save = SAR_img[idx].permute(1,2,0).detach().cpu().numpy()
                    NDVI_img_to_save = NDVI_img[idx].permute(1,2,0).detach().cpu().numpy()
                    np.save(os.path.join(save_path, "sar",  f'{unique_id}'), SAR_img_to_save)
                    np.save(os.path.join(save_path, "opt",  f'{unique_id}'), NDVI_img_to_save)

    def train(self, lr, epochs, check_preds_epoch, train_loader, val_loader, patience, loss, lr_scheduler=None):
        '''
        This function performs the training of the model, saves the snapshots at each check_preds_epoch epoch and at the end of the training.

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
        elif loss == "CLIP":
            mse_loss = nn.MSELoss()
            loss_function = CLIPLoss(self.vae_model, mse_loss, device=self.device, lambda_clip=0.5)
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
            for i,(SAR_img,NDVI_img) in enumerate(pbar_train):
                SAR_img = SAR_img.to(self.device)
                NDVI_img = NDVI_img.to(self.device)
                
                t = self.sample_timesteps(NDVI_img.shape[0]).to(self.device)
                # t is a unidimensional tensor of shape (NDVI_img.shape[0] that is the batch_size) with random integers from 1 to noise_steps.
                x_t, noise = self.noise_images(NDVI_img, t) # get the noisy images

                optimizer.zero_grad() # set the gradients to 0
                predicted_noise = model(x_t, t, SAR_img) 

                if loss == "CLIP":
                    x0_pred = self.xt_to_x0(x_t, t, predicted_noise)
                    train_loss, mse_loss, clip_loss = loss_function(predicted_noise, noise, NDVI_img, x0_pred)
                else:
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
                            self.prediction_plot(ema_model, train_loader, epoch)
                        else:
                            self._save_snapshot(epoch, model)
                            self.prediction_plot(model, train_loader, epoch)
                        
            else:
                if epoch % check_preds_epoch == 0:
                    if val_loader is None: # if there is no validation loader, then we save the weights at the frequency check_preds_epoch
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)
                            # self.prediction_plot(ema_model, train_loader, epoch)
                        else:
                            self._save_snapshot(epoch, model)
                            # self.prediction_plot(model, train_loader, epoch)

            if val_loader is not None:
                with torch.no_grad():
                    model.eval()
                    
                    for (SAR_img,NDVI_img) in pbar_val:
                        SAR_img = SAR_img.to(self.device)
                        NDVI_img = NDVI_img.to(self.device)

                        t = self.sample_timesteps(NDVI_img.shape[0]).to(self.device) # t is a unidimensional tensor of shape (NDVI_img.shape[0] that is the batch_size)with random integers from 1 to noise_steps.
                        x_t, noise = self.noise_images(NDVI_img, t) # get the noisy images
                        
                        if self.ema_smoothing:
                            predicted_noise = ema_model(x_t, t, SAR_img)
                        else:
                            predicted_noise = model(x_t, t, SAR_img) 
                        
                        if loss == "CLIP":
                            x0_pred = self.xt_to_x0(x_t, t, predicted_noise)
                            train_loss, mse_loss, clip_loss = loss_function(predicted_noise, noise, NDVI_img, x0_pred)
                        else:
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
    
    def prediction_plot(self, model, data_loader, epoch):
        fig, axs = plt.subplots(5,3, figsize=(15,15))
        for i in range(5):
            SAR_img = data_loader.dataset[i][0].to(self.device)
            NDVI_img = data_loader.dataset[i][1].to(self.device)

            NDVI_pred_img = self.sample(n=1,model=model, SAR_img=SAR_img, generate_video=False)
            
            axs[i,0].imshow(SAR_img[0].unsqueeze(0).permute(1,2,0).cpu().numpy())
            axs[i,0].set_title('SAR image')
            axs[i,1].imshow(NDVI_img.permute(1,2,0).cpu().numpy())
            axs[i,1].set_title('NDVI image')
            axs[i,2].imshow(NDVI_pred_img[0].permute(1,2,0).cpu().numpy())
            axs[i,2].set_title('NDVI pred image')

        plt.savefig(os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', f'NDVI_pred_{epoch}_epoch.png'))


class vae_loss(nn.Module):
    def __init__(self, device, lambda_rec=1.0, lambda_latent=0.5):
        super(vae_loss, self).__init__()
        self.lpips_loss = LPIPS(net='vgg').to(device)
        self.lambda_rec = lambda_rec
        self.lambda_latent = lambda_latent
        
    def forward(self, x_SAR, x_NDVI, latents_SAR, latents_NDVI, reconstructed_SAR, reconstructed_NDVI):
        rec_loss = self._reconstruction_loss(x_SAR, x_NDVI, reconstructed_SAR, reconstructed_NDVI)
        latent_loss = self._latent_consistency_loss(latents_SAR, latents_NDVI)
        total_loss = self.lambda_rec * rec_loss + self.lambda_latent * latent_loss
        return total_loss

    def _reconstruction_loss(self, x_SAR, x_NDVI, reconstructed_SAR, reconstructed_NDVI):
        rec_loss = (F.l1_loss(reconstructed_NDVI, x_NDVI) + F.l1_loss(reconstructed_SAR, x_SAR) +
            self.lpips_loss(reconstructed_NDVI, x_NDVI).mean())
        return rec_loss
    
    def _latent_consistency_loss(self, latents_SAR, latents_NDVI):
        '''
        The reason for this loss is that we want the latent space of the low resolution images
        to be similar to the latent space of the high resolution images. This is because the semantic
        information should be the same in both the low and high resolution images and in the latent 
        space we want all this information to be located and all the perceptual information to be
        discarded.
        '''
        latent_loss = F.mse_loss(latents_SAR, latents_NDVI)
        return latent_loss 

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

class CLIPLoss(nn.Module):
    def __init__(self, vae_model, mse_loss, device='cuda', lambda_clip=0.5):
        super(CLIPLoss, self).__init__()

        self.vae_model = vae_model
        self.device = device
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        self.clip_model = clip_model
        self.mse_loss = mse_loss
        self.lambda_clip = lambda_clip

    def clip_preprocess_tensor(self, image_tensor):
            # image_tensor: (B, 3, H, W) in [0, 1] range
            image_tensor = resize(image_tensor, [224, 224])
            image_tensor = (image_tensor - 0.48145466) / 0.26862954  # Normalize to CLIP range
            image_tensor = image_tensor.to(torch.float32)
            return image_tensor
    
    def forward(self, predicted_noise, noise, gt_latent, x0_pred):
        with torch.no_grad():
            decoded_pred = self.vae_model.decode(x0_pred)
            decoded_gt = self.vae_model.decode(gt_latent)
            # decoded_pred = decoded_pred.clamp(0,1)
            # decoded_gt = decoded_gt.clamp(0,1)
        decoded_pred = self.clip_preprocess_tensor(decoded_pred).repeat(1, 3, 1, 1)
        decoded_gt = self.clip_preprocess_tensor(decoded_gt).repeat(1, 3, 1, 1)
        with torch.no_grad():
            embed_pred = self.clip_model.encode_image(decoded_pred)
            embed_gt = self.clip_model.encode_image(decoded_gt)

        clip_loss = 1 - F.cosine_similarity(embed_pred, embed_gt).mean()

        mse = self.mse_loss(predicted_noise, noise)
        total_loss = mse + self.lambda_clip * clip_loss

        return total_loss, mse.item(), clip_loss.item()
    
def dataloader_PRE_encoding_maker(dataset_path, batch_size, multiple_gpus):
    train_path = f'{dataset_path}/train' 
    valid_path = f'{dataset_path}/test'

    train_dataset = get_data_SAR_TO_NDVI(train_path,SAR_channels=1,transform=None)
    val_dataset = get_data_SAR_TO_NDVI(valid_path,SAR_channels=1,transform=None)

    if multiple_gpus:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False, sampler=DistributedSampler(val_dataset))
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader

def dataloader_POST_encoding_maker(dataset_path, batch_size, multiple_gpus):
    global_min, global_max = compute_global_min_max(dataset_path)
    transform = GlobalMinMaxScaler(global_min, global_max)
    dataset = get_data_SAR_TO_NDVI(dataset_path, SAR_channels=4, data_format="numpy", transform=transform)
    if multiple_gpus:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset),drop_last=True)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader, global_min, global_max

def UNet_model_maker(UNet_type, SAR_channels, NDVI_channels, device):

    if UNet_type.lower() == 'residual attention unet':
        print('Using Residual Attention UNet')
        # model = Residual_Attention_UNet_SAR_TO_NDVI(SAR_channels, NDVI_channels, device).to(device)
        pass
    elif UNet_type.lower() == 'residual cross attention unet':
        print('Using Residual Cross Attention Nnet')
        model = Residual_CrossAttention_UNet_SAR_TO_NDVI(SAR_channels, NDVI_channels, device).to(device)
        pass
    elif UNet_type.lower() == 'residual visual multihead attention unet':
        print('Using Residual Visual MultiHead Attention UNet')
        pass
    else:
        raise ValueError('The UNet type must be either Residual Attention UNet or Residual MultiHead Attention UNet or Residual Visual MultiHeadAttention UNet')
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    return model

def VAE_model_maker(device, freeze_vae_params):
    vae_model_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(vae_model_path)
    vae_model = pipe.vae.to(device)
    vae_model = VAE_model_wrapped(vae_model, in_channels=1, freeze_vae_params=freeze_vae_params).to(device)
    return vae_model

def VAE_finetuning(dataset_path, image_size, batch_size, multiple_gpus, VAE_weight_path, device, freeze_vae_params):
    train_loader, val_loader = dataloader_PRE_encoding_maker(dataset_path=dataset_path, batch_size=batch_size, multiple_gpus=multiple_gpus)
    vae_model = VAE_model_maker(device, freeze_vae_params)
        
    if multiple_gpus:
        vae_model = DDP(vae_model, device_ids=[device], find_unused_parameters=True) 
    
    diffusion = Diffusion(
        noise_schedule=None, model=None, vae_model=vae_model,
        snapshot_path=None,
        VAE_weight_path=VAE_weight_path,
        noise_steps=None, beta_start=None, beta_end=None, device=device,
        image_size=image_size, model_name=None,
        multiple_gpus=multiple_gpus, ema_smoothing=None)
    
    if freeze_vae_params: # we freeze the parameters just to train the conv layer, before finetuning the VAE
        diffusion.fine_tuning_VAE(train_loader, epochs=15, learning_rate=1e-4) # epochs=15 for the Conv2d training, 50 for the VAE finetuning
    else:
        diffusion.fine_tuning_VAE(train_loader, epochs=50, learning_rate=1e-4)

        ########## ENCODE DATASET AND SAVE IT ##########
        encoded_images_train_save_path = os.path.join(dataset_path+'_VAE_encoded', "train")
        encoded_images_val_save_path = os.path.join(dataset_path+'_VAE_encoded', "val")
        if os.path.exists(os.path.join(encoded_images_train_save_path, 'img')):
            if len(os.listdir(os.path.join(encoded_images_train_save_path, 'img'))) == 0:
                diffusion.encoded_dataset_VAE(dataloader=train_loader, save_path=encoded_images_train_save_path)
                diffusion.encoded_dataset_VAE(dataloader=val_loader, save_path=encoded_images_val_save_path)
        else:
            diffusion.encoded_dataset_VAE(dataloader=train_loader, save_path=encoded_images_train_save_path)
            diffusion.encoded_dataset_VAE(dataloader=val_loader, save_path=encoded_images_val_save_path)

        ########## RENAME IMAGES (OPTIONAL) ##########
        # for set_path in [encoded_images_train_save_path, encoded_images_val_save_path]:
        #     for i, img_name in tqdm(enumerate(os.listdir(os.path.join(set_path, "img"))), desc='Renaming images', position=0):
        #         if img_name.endswith('.npy'):
        #                 os.rename(os.path.join(set_path, "img", img_name), os.path.join(set_path, "img", str(i+50000)+".npy"))

def Diffusion_training(snapshot_folder_path, model_name, snapshot_name,
                        noise_steps, ema_smoothing, UNet_type, SAR_channels, NDVI_channels,
                        VAE_weight_path, 
                                batch_size, image_size, multiple_gpus, 
                                    noise_schedule, dataset_path, lr,
                                     epochs,check_preds_epoch, patience,
                                      loss, lr_scheduler, device):

    os.makedirs(snapshot_folder_path, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(snapshot_folder_path), 'results'), exist_ok=True)

    model = UNet_model_maker(UNet_type, SAR_channels, NDVI_channels, device)

    if multiple_gpus:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)
    
    vae_model = VAE_model_maker(device, freeze_vae_params=True)
    diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model, vae_model=vae_model,
        snapshot_path=snapshot_path,
        VAE_weight_path=VAE_weight_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02 ,device=device,
        image_size=image_size, model_name=model_name,
        multiple_gpus=multiple_gpus, ema_smoothing=ema_smoothing)
        
    encoded_images_train_save_path = os.path.join(dataset_path, "train")
    encoded_images_val_save_path = os.path.join(dataset_path, "val")

    ########## CREATE DATALOADERS FOR THE POST-ENCODING MODEL ##########
    train_loader, global_min, global_max = dataloader_POST_encoding_maker(encoded_images_train_save_path, batch_size, multiple_gpus)
    # val_loader, global_min, global_max = dataloader_POST_encoding_maker(encoded_images_val_save_path, batch_size, multiple_gpus)
    np.save(os.path.join(os.path.dirname(snapshot_folder_path), "global_min.npy"),global_min)
    np.save(os.path.join(os.path.dirname(snapshot_folder_path), "global_max.npy"), global_max)
    val_loader = None
    ########## TRAIN DIFFUSION MODEL ##########
    diffusion.train(
        lr=lr, epochs=epochs, check_preds_epoch=check_preds_epoch,
        train_loader=train_loader, val_loader=val_loader, patience=patience, loss=loss,
        lr_scheduler=lr_scheduler)
    
    if multiple_gpus:
        destroy_process_group()

def sampling_test(noise_schedule, snapshot_folder_path, snapshot_name, VAE_weight_path, noise_steps, image_size, ema_smoothing,
                         UNet_type, model_name, generate_video, dataset_path, batch_size, SAR_channels, NDVI_channels, device='cuda'):

    snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

    vae_model = VAE_model_maker(device, freeze_vae_params=True)
    model = UNet_model_maker(UNet_type, SAR_channels, NDVI_channels, device)

    diffusion = Diffusion(
    noise_schedule=noise_schedule, model=model, vae_model=vae_model,
    snapshot_path=snapshot_path,
    VAE_weight_path=VAE_weight_path,
    noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, device=device,
    image_size=image_size, model_name=model_name,
    multiple_gpus=False, ema_smoothing=ema_smoothing)

    train_loader, val_loader = dataloader_PRE_encoding_maker(dataset_path=dataset_path, batch_size=batch_size, multiple_gpus=False)
    train_dataset = train_loader.dataset

    fig, axs = plt.subplots(5,3, figsize=(15,15))
    for i in range(5):
        SAR_img = train_dataset[i][0]
        NDVI_img = train_dataset[i][1]

        NDVI_pred_img = diffusion.sample(n=1,model=model, SAR_img=SAR_img, generate_video=generate_video)

        axs[i,0].imshow(SAR_img[0].unsqueeze(0).permute(1,2,0).cpu().numpy())
        axs[i,0].set_title('SAR image')
        axs[i,1].imshow(NDVI_img.permute(1,2,0).cpu().numpy())
        axs[i,1].set_title('NDVI image')
        axs[i,2].imshow(NDVI_pred_img[0].permute(1,2,0).cpu().numpy())
        axs[i,2].set_title('NDVI pred image')

    plt.savefig(os.path.join(os.getcwd(), 'models_run', model_name, 'results', 'SAR_TO_NDVI_results.png'))

def launch(args):
    '''
    This function is the main and call the training, the sampling and all the other functions in the Diffusion class.

    Input:
        image_size: the size of the images in the dataset
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
        SAR_channels: the number of SAR channels
        NDVI_channels: the number of NDVI channels
        generate_video: if True, the function will produce a video with the generated images
        loss: the loss function to use
        UNet_type: the type of UNet to use (Attention UNet, Residual Attention UNet)
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
    # SAR_channels, NDVI_channels = args.SAR_channels, args.NDVI_channels
    generate_video = args.generate_video
    loss = args.loss
    UNet_type = args.UNet_type
    multiple_gpus = args.multiple_gpus
    ema_smoothing = args.ema_smoothing
    VAE_weight_path = args.VAE_weight_path
    freeze_vae_params = args.freeze_vae_params

    
    if ema_smoothing:
        print(f'Using EMA smoothing')
    else:
        print(f'Not using EMA smoothing')

    if lr_scheduler and lr_scheduler.lower() != 'none':
        print(f'Using {lr_scheduler} learning rate scheduler')

    os.makedirs(snapshot_folder_path, exist_ok=True)
    os.makedirs(os.path.join(os.curdir, 'models_run', model_name, 'results'), exist_ok=True)
    
    if multiple_gpus:
        print('Using multiple GPUs')
        init_process_group(backend="nccl") # nccl stands for NVIDIA Collective Communication Library. It is used for distributed comunications across multiple GPUs.
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(int(device))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using single GPU')

    # VAE_finetuning(dataset_path=dataset_path, image_size=image_size,
    #                 batch_size=batch_size, multiple_gpus=multiple_gpus, 
    #                     VAE_weight_path=VAE_weight_path, device=device, freeze_vae_params=freeze_vae_params)
    
    # SAR images have 2 channels (VV and VH) and the NDVI image 1 channel. The VAE model we use need 3 channels in input. In order to make just few adjustments we use just the
    # first SAR channels (i.e. VV), and  we add a Conv2d layer that will convert the 1 channel of SAR and NDVI to 3 channels. So, SAR_channels=NDVI_channels=1.
    # Diffusion_training(snapshot_folder_path=snapshot_folder_path, model_name=model_name, snapshot_name=snapshot_name,
    #                     noise_steps=noise_steps, ema_smoothing=ema_smoothing,
    #                         UNet_type=UNet_type, SAR_channels=4, NDVI_channels=4, 
    #                         VAE_weight_path=VAE_weight_path,
    #                             batch_size=batch_size, image_size=image_size, multiple_gpus=multiple_gpus, 
    #                                 noise_schedule=noise_schedule, dataset_path=dataset_path, lr=lr,
    #                                  epochs=epochs, check_preds_epoch=check_preds_epoch, patience=patience,
    #                                   loss=loss, lr_scheduler=lr_scheduler, device=device)

    sampling_test(noise_schedule, snapshot_folder_path, snapshot_name, VAE_weight_path, noise_steps, image_size, ema_smoothing,
                         UNet_type, model_name, generate_video, dataset_path, batch_size, SAR_channels=4, NDVI_channels=4, device='cuda')

if __name__ == '__main__':
    import argparse  

    def str2bool(v):
        """Convert string to boolean."""
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default=None)
    parser.add_argument('--check_preds_epoch', type=int, default=20)
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--snapshot_name', type=str, default='snapshot.pt')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--noise_steps', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dataset_path', type=str, default=None)
    # parser.add_argument('--SAR_channels', type=int, default=2)
    # parser.add_argument('--NDVI_channels', type=int, default=1)
    parser.add_argument('--generate_video', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--UNet_type', type=str, default='Residual Attention UNet') # for now we have only the Residual Attention UNet
    parser.add_argument('--multiple_gpus', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--ema_smoothing', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--VAE_weight_path', type=str, default=None)
    parser.add_argument('--freeze_vae_params', type=str2bool, nargs='?', const=True, default=False)
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