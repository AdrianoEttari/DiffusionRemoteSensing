#!/bin/bash

##################### SUPER-RES #####################

# model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation"
# model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart"
# model_name="Residual_MultiHeadCrossAttention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart"
# model_name="Residual_MultiHeadCrossAttention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart_17mln"
# model_name="Residual_MultipleMultiHeadCrossAttention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart"
# model_name="Residual_MultipleMultiHeadCrossAttention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart"
model_name="Residual_MultipleMultiHeadCrossAttention_UNet_superres_magnification2_LRimgsize128_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart_MSELoss"

VAE_model_name="VAE_up42_LRandHR_finetuning_gradientAccumulation.pt"
# VAE_model_name="VAE_celeb100k_LRandHR_finetuning_gradientAccumulation.pt"

###### FINETUNING VAE ######
# dataset_path="up42_sentinel2_patches"
# dataset_path="celebA_10k"
# python train_diffusion_superres.py --dataset_path="$dataset_path" --model_name="$model_name" --Degradation_type="DownBlur" --image_size=256 --magnification_factor=4 --Blur_radius=0.5 --num_crops=1 --batch_size=4 --multiple_gpus=False --VAE_weight_path="$VAE_model_name" 

###### DIFFUSION TRAINING ######
# dataset_path="celebA_50k_VAE_encoded"
dataset_path="up42_sentinel2_patches_VAE_encoded"
python train_diffusion_superres.py --VAE_weight_path="$VAE_model_name" --model_name="$model_name" --snapshot_name=snapshot.pt --noise_steps=1000 --ema_smoothing=False --magnification_factor=2 --UNet_type="Residual Cross Attention UNet" --inp_out_channels=4 --batch_size=16 --image_size=256 --multiple_gpus=False --noise_schedule="cosine" --dataset_path="$dataset_path" --lr=1e-3 --epochs=201 --check_preds_epoch=5 --patience=10  --loss="MSE" --lr_schedule="cosine" --Degradation_type="DownBlur" --Blur_radius=0.5
# python train_diffusion_superres.py --model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart" --snapshot_name=snapshot.pt --noise_steps=1000 --ema_smoothing=False --magnification_factor=4 --UNet_type="Residual Cross Attention UNet" --inp_out_channels=4 --batch_size=32 --image_size=256 --multiple_gpus=False --noise_schedule="cosine" --dataset_path="celebA_50k_VAE_encoded" --lr=1e-3 --epochs=100 --check_preds_epoch=1 --patience=25  --loss="MSE" --lr_schedule="cosine"


###### SAMPLING ######
# dataset_path="celebA_100k"
# dataset_path="up42_sentinel2_patches"
# python train_diffusion_superres.py --model_name="$model_name" --snapshot_name=snapshot.pt --UNet_type="Residual Cross Attention UNet" --inp_out_channels=4 --image_size=256 --noise_schedule="cosine" --noise_steps=1000 --magnification_factor=2 --Degradation_type="DownBlur" --dataset_path="$dataset_path" --Blur_radius=0.5 --generate_video=False --VAE_weight_path="$VAE_model_name"
# python train_diffusion_superres.py --model_name="$model_name" --snapshot_name=snapshot.pt --UNet_type="Residual Cross Attention UNet" --inp_out_channels=4 --image_size=256 --noise_schedule="cosine" --noise_steps=1000 --magnification_factor=4 --Degradation_type="DownBlur" --dataset_path="$dataset_path" --Blur_radius=0.5 --num_crops=1 --batch_size=32 --generate_video=False --VAE_weight_path="$VAE_model_name"

###### SAMPLING (AGGREGATION SAMPLING) ######
# python Aggregation_Sampling.py --noise_schedule="cosine" --snapshot_name=snapshot.pt --image_size=256 --noise_steps=1000 --model_name="$model_name" --UNet_type="Residual Cross Attention UNet" --Degradation_type="DownBlur" --batch_dataloader_size=8 --magnification_factor=2 --inp_out_channels=4 --destination_path="rgb_60m_SR.png" --img_lr_path="rgb_60m.png" --VAE_weight_path="$VAE_model_name"



##################### SAR TO NDVI #####################

# model_name="Residual_MultipleMultiHeadCrossAttention_UNet_SAR_to_NDVI_StableDiffusion_gradientAccumulation_VAEapart"

# VAE_model_name="VAE_SAR_TO_NDVI_finetuning_gradientAccumulation.pt"

###### FINETUNING VAE ######
# dataset_path="SAR_TO_NDVI_dataset"
# python train_diffusion_SAR_TO_NDVI.py --epochs=15 --batch_size=6 --image_size=128 --lr=1e-4 --lr_scheduler="cosine" --check_preds_epoch=10 --noise_schedule="cosine" --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=1000 --patience=25 --dataset_path="$dataset_path" --generate_video=False --loss="CLIP" --UNet_type="Residual Cross Attention UNet" --multiple_gpus=False --ema_smoothing=False --VAE_weight_path="$VAE_model_name" --freeze_vae_params=False
# FINETUNING FOR 50 EPOCHS


###### DIFFUSION TRAINING ######
# dataset_path="SAR_TO_NDVI_dataset_VAE_encoded"
# python train_diffusion_SAR_TO_NDVI.py --epochs=15 --batch_size=2 --image_size=128 --lr=1e-4 --lr_scheduler="cosine" --check_preds_epoch=10 --noise_schedule="cosine" --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=10 --patience=25 --dataset_path="$dataset_path" --generate_video=False --loss="CLIP" --UNet_type="Residual Cross Attention UNet" --multiple_gpus=False --ema_smoothing=False --VAE_weight_path="$VAE_model_name" --freeze_vae_params=False


####### SAMPLING #######
# dataset_path="SAR_TO_NDVI_dataset"
# python train_diffusion_SAR_TO_NDVI.py --noise_schedule="cosine" --snapshot_name="snapshot.pt" --VAE_weight_path="$VAE_model_name" --noise_steps=1000 --image_size=128 --ema_smoothing=False --UNet_type="Residual Cross Attention UNet" --model_name="$model_name" --generate_video=False --dataset_path="$dataset_path" --batch_size=16
