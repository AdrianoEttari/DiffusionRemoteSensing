import torch
import os
from torchvision import transforms
from UNet_model_superres import Residual_Attention_UNet_superres
from UNet_model_superres_VMHA import Residual_Attention_UNet_superres_VMHA
from train_diffusion_superres import Diffusion as Diffusion_superres
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
import os
from UNet_model_SAR_TO_NDVI import Residual_Attention_UNet_SAR_TO_NDVI
from train_diffusion_SAR_TO_NDVI import Diffusion as Diffusion_SAR_TO_NDVI
import matplotlib.pyplot as plt

def super_resolver(lr_img, device, model_name, model='Residual Attention Unet'):
        '''
        This function takes a low resolution image, a device and a specific model_name and returns a super resolved image.
        Notice that the model_name must be formatted in a precise way. For example:
        'Residual_Attention_UNet_superres_magnification2_LRimgsize128_up42_sentinel2_patches_downblur'.
        The model_name parts must be separated by '_'. The model_name must contain the following information:
        - magnification
        - LRimgsize
        from these two informations (written exactly in this way) the function will extract the magnification
        factor and the low resolution image size.

        The function returns the super resolved image
        '''
        noise_schedule = 'cosine'
        noise_steps = 1500

        magnification_factor = int([info[13:] for info in model_name.split('_') if info.startswith('magnification')][0])
        image_size = int([info[9:] for info in model_name.split('_') if info.startswith('LRimgsize')][0]) * magnification_factor

        input_channels = output_channels = lr_img.shape[0]

        if model.lower() == 'residual attention unet':
                model = Residual_Attention_UNet_superres(input_channels, output_channels, device).to(device)
        elif model.lower() == 'residual vision multihead attention unet':
               model = Residual_Attention_UNet_superres_VMHA(input_channels, output_channels, image_size=image_size, device=device).to(device)

        snapshot_path = os.path.join('models_run', model_name, 'weights', 'snapshot.pt')

        print(f'HR Image size: {image_size}, LR Image size: {image_size//magnification_factor} Magnification factor: {magnification_factor}, Channels: {input_channels}')
        Degradation_type = 'DownBlur'

        diffusion = Diffusion_superres(
                noise_schedule=noise_schedule, model=model,
                snapshot_path=snapshot_path,
                noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02, 
                magnification_factor=magnification_factor,device=device,
                image_size=image_size, model_name=model_name, Degradation_type=Degradation_type)
        
        superres_img = diffusion.sample(n=1,model=model, lr_img=lr_img, input_channels=lr_img.shape[0], generate_video=False)
        superres_img = torch.clamp(superres_img, 0, 1)

        return superres_img

def plot_lr_sr(lr_img, sr_img, histogram=True, save_path=None):
        '''
        This function plots the low resolution image and the super resolved image.
        If histogram is set to True, the function will also plot the histograms of the three images.
        '''
        if histogram:
                fig, axs = plt.subplots(2,2, figsize=(15,10))
                title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 15}

                axs = axs.ravel()
                axs[0].imshow(lr_img.permute(1,2,0).detach().cpu())
                axs[0].set_title('low resolution image', fontdict=title_font)
                axs[1].imshow(sr_img[0].permute(1,2,0).detach().cpu())
                axs[1].set_title('super resolution image', fontdict=title_font)
                axs[2].hist(lr_img.flatten().detach().cpu(), bins=100)
                axs[2].set_title('lr image histogram', fontdict=title_font)
                axs[3].hist(sr_img.flatten().detach().cpu(), bins=100)
                axs[3].set_title('sr image histogram', fontdict=title_font)

                plt.show()
        else:
                fig, axs = plt.subplots(1,2, figsize=(15,10))
                title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 15}
                axs = axs.ravel()
                axs[0].imshow(lr_img.permute(1,2,0).detach().cpu())
                axs[0].set_title('low resolution image', fontdict=title_font)
                axs[1].imshow(sr_img[0].permute(1,2,0).detach().cpu())
                axs[1].set_title('super resolution image', fontdict=title_font)
                if save_path is not None:
                        plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.show()

def SAR_to_NDVI_generator(SAR_img_path, device, n_generations=1):
        '''
        This function performs the generation of NDVI images from SAR images. It uses the only model available for this task.
        model_name = 'Residual_Attention_UNet_EMA_imgsize128_SAR_TO_NDVI'
        By default it generates only one NDVI image. If you want to generate more than one, you can specify the number of generations.

        The function returns the generated NDVI images.
        '''
        noise_schedule = 'cosine'
        noise_steps = 1500
        SAR_channels=2
        NDVI_channels=1
        model_name = 'Residual_Attention_UNet_EMA_imgsize128_SAR_TO_NDVI'
        model = Residual_Attention_UNet_SAR_TO_NDVI(SAR_channels, NDVI_channels, device).to(device)
        snapshot_path = os.path.join('models_run', model_name, 'weights', 'snapshot.pt')
        image_size = int([info[7:] for info in model_name.split('_') if info.startswith('imgsize')][0])

        print(f'Image size: {image_size}, SAR channels: {SAR_channels}, NDVI channels: {NDVI_channels}')

        # Preprocessing
        SAR_img = torch.load(SAR_img_path)
        if SAR_img.min() < 0 and SAR_img.min() > -1:
                SAR_img = (SAR_img + 1) / 2
        elif SAR_img.min() < -1 or SAR_img.max() > 1:
               raise ValueError('SAR image values are not in the range [-1, 1]')
        
        diffusion = Diffusion_SAR_TO_NDVI(
                noise_schedule=noise_schedule, model=model,
                snapshot_path=snapshot_path,
                noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02,device=device,
                image_size=image_size, model_name=model_name,
                multiple_gpus=False, ema_smoothing=False)
        
        NDVI_pred_img = diffusion.sample(n=n_generations,model=model, SAR_img=SAR_img, NDVI_channels=NDVI_channels, generate_video=False)
        return NDVI_pred_img

def plot_SAR_NDVI(SAR_img, NDVI_img, NDVI_pred_img, save_path=None):
    '''
    This function plots the SAR image, the NDVI image and the generated NDVI images.
    If save_path is not None, the function will save the plot in the specified path.
    '''
    num_pred_imgs = NDVI_pred_img.shape[0]
    fig, axs = plt.subplots(1, 3 + num_pred_imgs, figsize=(5 * (2 + num_pred_imgs), 10))
    title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 8}
    axs = axs.ravel()

    axs[0].imshow(SAR_img[0].unsqueeze(0).permute(1, 2, 0).detach().cpu())
    axs[0].set_title('SAR image 1', fontdict=title_font)
    axs[0].axis('off')
    
    axs[1].imshow(SAR_img[1].unsqueeze(0).permute(1, 2, 0).detach().cpu())
    axs[1].set_title('SAR image 2', fontdict=title_font)
    axs[1].axis('off')
    
    axs[2].imshow(NDVI_img.permute(1, 2, 0).detach().cpu())
    axs[2].set_title('NDVI image', fontdict=title_font)
    axs[2].axis('off')
    
    for i in range(num_pred_imgs):
        axs[3 + i].imshow(NDVI_pred_img[i].permute(1, 2, 0).detach().cpu())
        axs[3 + i].set_title(f'NDVI pred {i}', fontdict=title_font)
        axs[3 + i].axis('off')
    
    plt.show()
    if save_path is not None:
        img_to_save = int(input('Which NDVI pred you want to save? (0, 1, 2, ...): '))
        fig, axs = plt.subplots(1, 4, figsize=(5, 10))
        axs = axs.ravel()       
        title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 8}
        axs[0].imshow(SAR_img[0].unsqueeze(0).permute(1, 2, 0).detach().cpu())
        axs[0].set_title('SAR image 1', fontdict=title_font)
        axs[0].axis('off')
        axs[1].imshow(SAR_img[1].unsqueeze(0).permute(1, 2, 0).detach().cpu())
        axs[1].set_title('SAR image 2', fontdict=title_font)
        axs[1].axis('off')
        axs[2].imshow(NDVI_img.permute(1, 2, 0).detach().cpu())
        axs[2].set_title('NDVI image', fontdict=title_font)
        axs[2].axis('off')
        axs[3].imshow(NDVI_pred_img[img_to_save].permute(1, 2, 0).detach().cpu())
        axs[3].set_title(f'NDVI pred', fontdict=title_font)
        axs[3].axis('off')
        plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight', pad_inches=0)
    

    
if __name__ == '__main__':
        #### SUPER RESOLUTION EXAMPLE ####
        # device = 'mps'
        # img_path = os.path.join('assets','Other','up42_sample_lr.png')
        # to_tensor = transforms.ToTensor()
        # lr_img = to_tensor(Image.open(img_path)).to(device)
        # model_name = 'Residual_Attention_UNet_superres_magnification2_LRimgsize128_up42_sentinel2_patches_downblur'
        # superres_img = super_resolver(lr_img, device, model_name,model='Residual Attention Unet')
        # file_name = os.path.basename(img_path)
        # save_path = os.path.join('assets','Other',file_name.replace('lr', 'sr'))
        # plot_lr_sr(lr_img, superres_img, histogram=False, save_path=save_path)


        #### SUPER RESOLUTION EXAMPLE ####
        device = 'mps'
        img_path = os.path.join('celebA_10k','test_original','000114.jpg')
        img = Image.open(img_path)
        img = img.resize((192,192))
        downsample = transforms.Resize((img.size[0] // 4, img.size[1] // 4),
                                       interpolation=transforms.InterpolationMode.BICUBIC)
        img = downsample(img)

        to_tensor = transforms.ToTensor()
        lr_img = to_tensor(img).to(device)

        model_name = 'Residual_VisionMultiHeadAttention_UNet_superres_magnification4_LRimgsize48_CelebA100k_downblur_4'
        superres_img = super_resolver(lr_img, device, model_name,model='Residual Vision Multihead Attention Unet')
        file_name = os.path.basename(img_path)
        plot_lr_sr(lr_img, superres_img, histogram=False)

        #### SAR TO NDVI EXAMPLE ####
        # device = 'mps'
        # test_path = os.path.join('imgs_sample', 'test_SAR_TO_NDVI')
        # list_of_files = ['Victoria_0_20210830_patch_289.pt']
        # SAR_img_path = os.path.join(test_path, 'sar', list_of_files[0])
        # SAR_img = torch.load(SAR_img_path)
        # NDVI_img = torch.load(os.path.join(test_path, 'opt', list_of_files[0]))

        # NDVI_pred_img = SAR_to_NDVI_generator(SAR_img_path, device, n_generations=5)
        # destination_path = 'models_run/Residual_Attention_UNet_EMA_imgsize128_SAR_TO_NDVI/results'
        # save_path = os.path.join(destination_path, f'{list_of_files[0].replace(".pt", ".png")}')
        # plot_SAR_NDVI(SAR_img, NDVI_img, NDVI_pred_img, save_path=save_path)


