#!/bin/bash
# model_name="TO_REMOVE_generate"
# python3 train_diffusion_generation.py --epochs=1 --noise_schedule="cosine" --batch_size=5 --image_size=32 --lr=2e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --patience=25  --dataset_path="ImageNet256_small" --inp_out_channels=3 --loss="MSE" --UNet_type="DiffiT UNet" --multiple_gpus="False" --ema_smoothing="False" 

model_name="Residual_MultipleMultiHeadCrossAttention_UNet_generation_102flowers"

VAE_model_name="VAE_102flowers_finetuning_gradientAccumulation.pt"

###### FINETUNING VAE ######
# dataset_path="102flowers_dataset"
# python train_diffusion_generation.py --dataset_path="$dataset_path" --image_size=512 --batch_size=1 --multiple_gpus=False --VAE_weight_path="$VAE_model_name" 

###### DIFFUSION TRAINING ######
dataset_path="102flowers_dataset_VAE_encoded"
python train_diffusion_generation.py --epochs=2 --batch_size=1 --image_size=64 --lr=1e-4 --lr_schedule="cosine" --check_preds_epoch=10 --noise_schedule="cosine" --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --patience=25  --dataset_path="$dataset_path" --inp_out_channels=4 --generate_video="False" --loss="MSE" --UNet_type="Residual Cross Attention Unet" --multiple_gpus="False" --ema_smoothing="False" --VAE_weight_path="$VAE_model_name" 

###### SAMPLING ######
# dataset_path="102flowers_dataset"
# python train_diffusion_generation.py --noise_schedule="cosine" --snapshot_name=snapshot.pt --VAE_weight_path="$VAE_model_name" --noise_steps=1000 --image_size=512 --ema_smoothing=False --UNet_type="Residual Cross Attention UNet" --model_name="$model_name" --generate_video=False --num_classes=102
# python train_diffusion_superres.py --model_name="$model_name" --snapshot_name=snapshot.pt --UNet_type="Residual Cross Attention UNet" --inp_out_channels=4 --image_size=256 --noise_schedule="cosine" --noise_steps=1000 --magnification_factor=4 --Degradation_type="DownBlur" --dataset_path="$dataset_path" --Blur_radius=0.5 --num_crops=1 --batch_size=32 --generate_video=False --VAE_weight_path="$VAE_model_name"
# python train_diffusion_superres.py --model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_celeb50k_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation_VAEapart" --snapshot_name=snapshot.pt --UNet_type="Residual Attention UNet" --inp_out_channels=4 --image_size=256 --noise_schedule="cosine" --noise_steps=1000 --magnification_factor=4 --Degradation_type="DownBlur" --dataset_path="celebA_100k" --Blur_radius=0.5 --num_crops=1 --batch_size=32 --generate_video=False --VAE_weight_path="VAE_celeb100k_LRandHR_finetuning_gradientAccumulation"

###### SAMPLING (AGGREGATION SAMPLING) ######
# python Aggregation_Sampling.py --noise_schedule="cosine" --snapshot_name=snapshot.pt --image_size=256 --noise_steps=1000 --model_name="Residual_Attention_UNet_superres_magnification4_LRimgsize64_up42_sentinel2_patches_downblur_StableDiffusion_LRandHR_gradientAccumulation" --UNet_type="Residual Attention UNet" --Degradation_type="DownBlur" --batch_dataloader_size=8 --magnification_factor=4 --inp_out_channels=4 --destination_path="rgb_60m_SR.png" --img_lr_path="rgb_60m.png" --VAE_weight_path="VAE_up42_LRandHR_finetuning_gradientAccumulation"
