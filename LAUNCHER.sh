#!/bin/bash
# model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation"
model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart"

# VAE_model_name_HR="VAE_up42_hr256_finetuning"
# VAE_model_name_LR="VAE_up42_lr64_finetuning"
# VAE_model_name="VAE_up42_LRandHR_finetuning_gradientAccumulation"
VAE_model_name="VAE_celeb100k_LRandHR_finetuning_gradientAccumulation"

###### FINETUNING VAE ######
# dataset_path="up42_sentinel2_patches"
# dataset_path="celebA_10k"
# python train_diffusion_superres.py --dataset_path="$dataset_path" --Degradation_type="DownBlur" --image_size=256 --magnification_factor=4 --Blur_radius=0.5 --num_crops=1 --batch_size=4 --multiple_gpus=False --VAE_weight_path="$VAE_model_name" 

###### DIFFUSION TRAINING ######
# dataset_path="celebA_50k_VAE_encoded"
# python train_diffusion_superres.py --model_name="$model_name" --snapshot_name=snapshot.pt --noise_steps=1000 --ema_smoothing=False --magnification_factor=4 --UNet_type="Residual Attention UNet" --inp_out_channels=4 --batch_size=32 --image_size=256 --multiple_gpus=False --noise_schedule="cosine" --dataset_path="$dataset_path" --lr=1e-3 --epochs=100 --check_preds_epoch=1 --patience=25  --loss="MSE" --lr_schedule="cosine"
# python train_diffusion_superres.py --model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart" --snapshot_name=snapshot.pt --noise_steps=1000 --ema_smoothing=False --magnification_factor=4 --UNet_type="Residual Attention UNet" --inp_out_channels=4 --batch_size=32 --image_size=256 --multiple_gpus=False --noise_schedule="cosine" --dataset_path="celebA_50k_VAE_encoded" --lr=1e-3 --epochs=100 --check_preds_epoch=1 --patience=25  --loss="MSE" --lr_schedule="cosine"

###### SAMPLING ######
# dataset_path="celebA_10k"
# python train_diffusion_superres.py --model_name="$model_name" --snapshot_name=snapshot.pt --UNet_type="Residual Attention UNet" --inp_out_channels=4 --image_size=256 --noise_schedule="cosine" --noise_steps=1000 --magnification_factor=4 --Degradation_type="DownBlur" --dataset_path="$dataset_path" --Blur_radius=0.5 --num_crops=1 --batch_size=32 --generate_video=False --VAE_weight_path="$VAE_model_name"
# python train_diffusion_superres.py --model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart" --snapshot_name=snapshot.pt --UNet_type="Residual Attention UNet" --inp_out_channels=4 --image_size=256 --noise_schedule="cosine" --noise_steps=1000 --magnification_factor=4 --Degradation_type="DownBlur" --dataset_path="celebA_100k" --Blur_radius=0.5 --num_crops=1 --batch_size=32 --generate_video=False --VAE_weight_path="VAE_celeb100k_LRandHR_finetuning_gradientAccumulation"

###### SAMPLING (AGGREGATION SAMPLING) ######
python Aggregation_Sampling.py --noise_schedule="cosine" --snapshot_name=snapshot.pt --image_size=256 --noise_steps=1000 --model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation" --UNet_type="Residual Attention UNet" --Degradation_type="DownBlur" --magnification_factor=4 --inp_out_channels=4 --destination_path="rgb_20m_SR.png" --img_lr_path="rgb_20m.png" --VAE_weight_path="VAE_up42_LRandHR_finetuning_gradientAccumulation"

