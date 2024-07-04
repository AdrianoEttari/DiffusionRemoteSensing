import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
from utils import video_maker
import numpy as np
import torchvision

from UNet_model_generation import Residual_Attention_UNet_generation,EMA

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # By using .features, we are considering just the convolutional part of the VGG network
        # without considering the avg pooling and the fully connected layers which map the features to the 1000 classes
        # and so, solve the classification task.
        # self.vgg = nn.Sequential(*[self.vgg[i] for i in range(8)]) 
        self.vgg.to(device)
        self.vgg.eval()  # Set VGG to evaluation mode
        self.device = device
        # Freeze all VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
    def preprocess_image(self, image):
        '''
        The VGG network wants input sized (224,224), normalized and as pytorch tensor. 
        '''
        transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        if image.shape[-1] != 224:
            try:
                image = F.interpolate(image, size=(224, 224), mode='bicubic', align_corners=False)
            except:
                image = F.interpolate(image.to('cpu'),  size=(224, 224), mode='bicubic', align_corners=False).to(self.device)
        
        return transform(image)
    
    def forward(self, x, y):
        x = self.preprocess_image(x)
        y = self.preprocess_image(y)

        x_features = self.vgg(x)
        y_features = self.vgg(y)

        return torch.mean((x_features-y_features)**2)

class CombinedLoss(nn.Module):
    def __init__(self, first_loss, second_loss, weight_first=0.5):
        super(CombinedLoss, self).__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.weight_first = weight_first  

    def forward(self, predicted, target):
        first_loss_value = self.first_loss(predicted, target)
        second_loss_value = self.second_loss(predicted, target)
        combined_loss = self.weight_first * first_loss_value + (1-self.weight_first) * second_loss_value
        return combined_loss
    
class Diffusion:
    def __init__(
            self,
            noise_schedule: str,
            model: nn.Module,
            snapshot_path: str,
            noise_steps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            device='cuda',
            image_size=224,
            model_name='generation',
            multiple_gpus=False,
            ema_smoothing=False,
            ):
    
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.model_name = model_name
        self.device = device
        self.snapshot_path = snapshot_path
        self.multiple_gpus = multiple_gpus
        self.ema_smoothing = ema_smoothing
        self.model = model.to(self.device)
        # epoch_run is used by _save_snapshot and _load_snapshot to keep track of the current epoch
        self.epochs_run = 0
        # If a snapshot exists, we load it
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()

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
        # if a tensor has shape (2,3,4), a[None,None,:,None] will be shaped (1,1,2,1,3,4)).Basically, the dimensions are added where the None
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
    
    def sample(self, n, model, target_class=None, cfg_scale=3, input_channels=3, generate_video=False):
        '''
        As the name suggests this function is used for sampling. Therefore we want to 
        loop backward (moreover, notice that in the sample we want to perform EVERY STEP CONTIGUOUSLY
        while at training time we use the sample_timesteps() function to get just one random time step per batch).

        What we do is to predict the noise conditionally, then if the cfg_scale is > 0,
        we also predict the noise unconditionally. Eventually, we apply the formula
        out of the CFG paper using the torch.lerp function which does exactly the same

        predicted_noise = uncond_predicted_noise + cfg_scale * (predicted_noise - uncond_predicted_noise)

        and this will be our final predicted noise.

        Input:
            n: the number of images we want to sample
            target_class: the target class for the images
            cfg_scale: the scale of the CFG noise
            input_channels: the number of input channels
            generate_video: if True, the function will produce a video with the generated NDVI images.
        
        Output:
            x: a tensor of shape (n, input_channels, self.image_size, self.image_size) with the generated images
        '''

        frames = []
        model.eval() # disables dropout and batch normalization
        with torch.no_grad(): # disables gradient calculation
            x = torch.randn((n, input_channels, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): 
                t = (torch.ones(n) * i).long().to(self.device) # tensor of shape (n) with all the elements equal to i.
                # Basically, each of the n image will be processed with the same integer time step t.
                
                predicted_noise = model(x, t, target_class)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale) # Compute the formula of the CFG paper https://arxiv.org/abs/2207.12598

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    # If i>1 then we add noise to the image we have sampled (remember that from x_t we sample x_{t-1}).
                    # If i==1 we sample x_0, which is the final image we want to generate, so we don't add noise.
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) # we don't add noise in the last time step because it would just make the final outcome worse.
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if generate_video == True:
                    frames.append(x)
        if generate_video == True:
            video_maker(frames, os.path.join(os.getcwd(), 'models_run', self.model_name, 'results', 'video_denoising.mp4'), 100)
        model.train() # enables dropout and batch normalization
        return x

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

            snapshot = torch.load(self.snapshot_path, map_location='cpu')
            model_state = OrderedDict((key.replace('module.', ''), value) for key, value in snapshot['MODEL_STATE'].items())
            self.model.module.load_state_dict(model_state)
            self.model.module.to(self.device)
        else:
            snapshot = torch.load(self.snapshot_path, map_location=self.device)
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.epochs_run = snapshot["EPOCHS_RUN"]

        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def early_stopping(self, patience, epochs_without_improving):
        '''
        This function checks if the validation loss is increasing. If it is for more than patience times,
        then it returns True (that will correspond to breaking the training loop).
        '''
        if epochs_without_improving >= patience:
            print('Early stopping! Training stopped')
            return True

    def train(self, lr, epochs, check_preds_epoch, train_loader, val_loader, patience, loss, verbose):
        '''
        This function performs the training of the model, saves the snapshots and the model at the end of the training each self.every_n_epochs epochs.

        Input:
            lr: the learning rate
            epochs: the number of epochs
            check_preds_epoch: the frequency at which the model will save the predictions
            train_loader: the training loader
            val_loader: the validation loader
            patience: the number of epochs after which the training will be stopped if the validation loss is increasing
            loss: the loss function to use
            verbose: if True, the function will use the tqdm during the training and the validation
        '''
        num_classes = len(train_loader.dataset.classes)
        model = self.model

        if self.ema_smoothing:
            ema = EMA(beta=0.995)############################# EMA ############################
            ema_model = copy.deepcopy(model).eval().requires_grad_(False)############################# EMA ############################

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW is a variant of Adam that adds weight decay (L2 regularization)
        # Basically, weight decay is a regularization technique that penalizes large weights. It's a way to prevent overfitting. In AdamW, 
        # the weight decay is added to the gradient and not to the weights. This is because the weights are updated in a different way in AdamW.

        if loss == 'MSE':
            loss_function = nn.MSELoss()
        elif loss == 'MAE':
            loss_function = nn.L1Loss()
        elif loss == 'Huber':
            loss_function = nn.HuberLoss() 
        elif loss == 'MSE+Perceptual_noise':
            vgg_loss = VGGPerceptualLoss(self.device)
            mse_loss = nn.MSELoss()
            loss_function = CombinedLoss(first_loss=mse_loss, second_loss=vgg_loss, weight_first=0.3)
        else:
            raise ValueError('The Loss must be either MSE or MAE or Huber or MSE+Perceptual_noise')
               
        epochs_without_improving = 0
        best_loss = float('inf')  

        for epoch in range(self.epochs_run, epochs):
            if self.multiple_gpus:
                train_loader.sampler.set_epoch(epoch) # ensures that the data is shuffled in a consistent manner across multiple epochs (it is useful just for the DistributedSampler)
            if verbose:
                pbar_train = tqdm(train_loader,desc='Training', position=0)
                if val_loader is not None:
                    pbar_val = tqdm(val_loader,desc='Validation', position=0)
            else:
                pbar_train = train_loader
                if val_loader is not None:
                    pbar_val = val_loader

            running_train_loss = 0.0
            running_val_loss = 0.0
            
            model.train()
            for i, (img,label) in enumerate(pbar_train):
                label = label.to(self.device)
                img = img.to(self.device)

                t = self.sample_timesteps(img.shape[0]).to(self.device)
                # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size) with random integers from 1 to noise_steps.
                x_t, noise = self.noise_images(img, t) # get batch_size noise images

                optimizer.zero_grad() # set the gradients to 0

                if np.random.random() < 0.1: # 10% of the time, don't pass labels (we train 10% of the times uncoditionally and 90% conditionally)
                    label = None
                predicted_noise = model(x_t, t, label) # here we pass the plain labels to the model (e.g. 0,1,2,...,9 if there are 10 classes)
                
                train_loss = loss_function(predicted_noise, noise)
                
                train_loss.backward() # compute the gradients
                optimizer.step() # update the weights
                
                if self.ema_smoothing:
                    ema.step_ema(ema_model, model)############################# EMA ############################

                if verbose:
                    pbar_train.set_postfix(LOSS=train_loss.item()) # set_postfix just adds a message or value displayed after the progress bar. In this case the loss of the current batch.
            
                running_train_loss += train_loss.item()

            running_train_loss /= len(train_loader) # at the end of each epoch I want the average loss
            print(f"Epoch {epoch}: Running Train ({loss}) {running_train_loss}")

            # num_classes = None
            # IF THERE ARE MULTIPLE GPUs, MAKE JUST THE FIRST ONE SAVE THE SNAPSHOT AND COMPUTE THE PREDICTIONS TO AVOID REDUNDANCY
            # IN THE ELSE STATEMENT, THERE IS EXACTLY THE SAME. 
            if self.multiple_gpus:
                if self.device==0 and epoch % check_preds_epoch == 0:
                    if val_loader is None:
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)############################# EMA ############################
                        else:
                            self._save_snapshot(epoch, model)

                    fig, axs = plt.subplots(num_classes,5, figsize=(15,15))
                    for i in range(num_classes):
                        if self.ema_smoothing:
                            prediction = self.sample(n=5,model=ema_model, target_class=torch.tensor([i], dtype=torch.int64).to(self.device), input_channels=train_loader.dataset[0][0].shape[0], generate_video=False)
                        else:
                            prediction = self.sample(n=5,model=model, target_class=torch.tensor([i], dtype=torch.int64).to(self.device), input_channels=train_loader.dataset[0][0].shape[0], generate_video=False)
                        for j in range(5):
                            axs[i,j].imshow(prediction[j].permute(1,2,0).cpu().numpy())
                            axs[i,j].set_title(f'Class {i}')

                    plt.savefig(os.path.join('..', 'models_run', self.model_name, 'results', f'generation_{epoch}_epoch.png'))
            else:
                if epoch % check_preds_epoch == 0:
                    if val_loader is None: # if there is no validation loader, then we save the weights at the frequency check_preds_epoch
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)############################# EMA ############################
                        else:
                            self._save_snapshot(epoch, model)

                    fig, axs = plt.subplots(num_classes,5, figsize=(15,15))
                    for i in range(num_classes):
                        if self.ema_smoothing:
                            prediction = self.sample(n=5,model=ema_model, target_class=torch.tensor([i], dtype=torch.int64).to(self.device), input_channels=train_loader.dataset[0][0].shape[0], generate_video=False)
                        else:
                            prediction = self.sample(n=5,model=model, target_class=torch.tensor([i], dtype=torch.int64).to(self.device), input_channels=train_loader.dataset[0][0].shape[0], generate_video=False)
                        for j in range(5):
                            axs[i,j].imshow(prediction[j].permute(1,2,0).cpu().numpy())
                            axs[i,j].set_title(f'Class {i}')

                    plt.savefig(os.path.join('..', 'models_run', self.model_name, 'results', f'generation_{epoch}_epoch.png'))

            if val_loader is not None:
                with torch.no_grad():
                    model.eval()
                    
                    for (img,label) in pbar_val:
                        label = label.to(self.device)
                        img = img.to(self.device)

                        t = self.sample_timesteps(img.shape[0]).to(self.device)
                        # t is a unidimensional tensor of shape (images.shape[0] that is the batch_size)with random integers from 1 to noise_steps.
                        x_t, noise = self.noise_images(img, t) # get batch_size noise images
                        
                        if np.random.random() < 0.1: # 10% of the time, don't pass labels (we train 10% of the times uncoditionally and 90% conditionally)
                            label = None
                        if self.ema_smoothing:
                            predicted_noise = ema_model(x_t, t, label) ############################# EMA ############################
                        else:
                            predicted_noise = model(x_t, t, label) # here we pass the plain labels to the model (i.e. 0,1,2,...,9 if there are 10 classes)

                        val_loss = loss_function(predicted_noise, noise)

                        if verbose:
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
                                self._save_snapshot(epoch, ema_model)############################# EMA ############################
                            else:
                                self._save_snapshot(epoch, model)
                    else:
                        if self.ema_smoothing:
                            self._save_snapshot(epoch, ema_model)############################# EMA ############################
                        else:
                            self._save_snapshot(epoch, model)  
                else:
                    epochs_without_improving += 1

                if self.early_stopping(patience, epochs_without_improving):
                    break
            print('Epochs without improving: ', epochs_without_improving)

def launch(args):
    '''
    This function is the main and call the training, the sampling and all the other functions in the Diffusion class.

    Input:
        image_size: the size of the images in the dataset
        dataset_path: the path of the dataset
        batch_size: the batch size
        lr: the learning rate
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
        loss: the loss function to use
        UNet_type: the type of UNet to use (residual attention unet)
        multiple_gpus: if True, the function will use multiple GPUs
        ema_smoothing: if True, the function will use EMA smoothing

    Output:
        None
    '''
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    lr = args.lr
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
    loss = args.loss
    UNet_type = args.UNet_type
    multiple_gpus = args.multiple_gpus
    ema_smoothing = args.ema_smoothing

    if ema_smoothing:
        print('Using EMA smoothing')
    else:
        print('Not using EMA smoothing')
    
    os.makedirs(snapshot_folder_path, exist_ok=True)
    os.makedirs(os.path.join('..', 'models_run', model_name, 'results'), exist_ok=True)
    
    if multiple_gpus:
        print('Using multiple GPUs')
        init_process_group(backend="nccl") # nccl stands for NVIDIA Collective Communication Library. It is used for distributed comunications across multiple GPUs.
    else:
        print('Using a single GPU')

    if dataset_path.lower() == 'cifar10':
        transform = transforms.Compose(
        [transforms.ToTensor()])
        train_dataset = torchvision.datasets.CIFAR10(root='./Cifar10', train=True,
                                            download=True, transform=transform)
        image_size = 32
    else:
        image_size = args.image_size
        transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
            ]) 
        train_path = f'../{dataset_path}'
        train_dataset = datasets.ImageFolder(train_path, transform=transform)

    if multiple_gpus:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(train_loader.dataset.classes)

    if multiple_gpus:
        gpu_id = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(int(gpu_id))
        device = gpu_id
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if UNet_type.lower() == 'residual attention unet':
        print('Using Residual Attention UNet')
        model = Residual_Attention_UNet_generation(input_channels, output_channels, num_classes, device).to(device)
    elif UNet_type.lower() == 'residual multihead attention unet':
        print('Using Residual MultiHead Attention UNet')
        # model = Residual_MultiHeadAttention_UNet_superres(input_channels, output_channels, device).to(device)
        pass
    elif UNet_type.lower() == 'residual visual multihead attention unet':
        print('Using Residual Visual MultiHead Attention UNet')
        # model = Residual_Visual_MultiHeadAttention_UNet_superres(input_channels, image_size ,output_channels, device).to(device)
        pass
    else:
        raise ValueError('The UNet type must be either Residual Attention UNet or Residual MultiHead Attention UNet or Residual Visual MultiHeadAttention UNet superres')
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    if multiple_gpus:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    snapshot_path = os.path.join(snapshot_folder_path, snapshot_name)

    diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02,
        device=device, image_size=image_size, model_name=model_name,
        multiple_gpus=multiple_gpus, ema_smoothing=ema_smoothing)

    # Training 
    diffusion.train(
        lr=lr, epochs=epochs, check_preds_epoch=check_preds_epoch,
        train_loader=train_loader, val_loader=None, patience=patience, loss=loss, verbose=True)
    
    if multiple_gpus:
        destroy_process_group()

    fig, axs = plt.subplots(num_classes,5, figsize=(15,15))

    for i in range(num_classes):
        prediction = diffusion.sample(n=5,model=model, target_class=torch.tensor([i], dtype=torch.int64).to(device), input_channels=train_loader.dataset[0][0].shape[0], generate_video=generate_video)
        for j in range(5):
            axs[i,j].imshow(prediction[j].permute(1,2,0).cpu().numpy())
            axs[i,j].set_title(f'Class {i}')

    plt.savefig(os.path.join('..', 'models_run', model_name, 'results', f'generation_results.png'))


if __name__ == '__main__':
    import argparse     

    def str2bool(v):
        """Convert string to boolean."""
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--check_preds_epoch', type=int, default=20)
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--snapshot_name', type=str, default='snapshot.pt')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--noise_steps', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--inp_out_channels', type=int, default=3) # input channels must be the same of the output channels
    parser.add_argument('--generate_video',  type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--UNet_type', type=str, default='Residual Attention UNet') # for now we have only the Residual Attention UNet
    parser.add_argument('--multiple_gpus', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--ema_smoothing', type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    args.snapshot_folder_path = os.path.join('..', 'models_run', args.model_name, 'weights')
    launch(args)