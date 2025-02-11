#!/bin/bash
model_name="TO_REMOVE"
VAE_model_name_hr="VAE_up42_lr64_finetuning"

# python3 train_diffusion_superres_DIFFUSERS.py --epochs=1 --noise_schedule="cosine" --batch_size=5 --image_size=192 --lr=1e-4 --snapshot_name=snapshot.pt --model_name="TO_REMOVE" --noise_steps=2 --patience=25  --dataset_path="celebA_100k" --inp_channels=6 --loss="MSE" --magnification_factor=4 --Degradation_type="DownBlur" --multiple_gpus="False" --Blur_radius=0.5

# python3 train_diffusion_superres.py --epochs=1 --batch_size=5 --image_size=256 --lr=1e-4 --lr_scheduler="cosine"--noise_schedule="cosine" --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --patience=25  --dataset_path="up42_sentinel2_patches" --inp_out_channels=3 --loss="MSE" --magnification_factor=2 --UNet_type="DiffiT UNet" --Degradation_type="DownBlur" --multiple_gpus="False" --ema_smoothing="False" --Blur_radius=0.5 

# dataset_path="celebA_100k"
# python3 StableDiff_LatentDiffusion.py --epochs=10 --batch_size=8 --image_size=192 --lr=1e-5 --dataset_path="$dataset_path" --magnification_factor=4 --Degradation_type="DownBlur" --multiple_gpus="False" --Blur_radius=0.5
# python StableDiff_LatentDiffusion.py --epochs=10  --batch_size=2 --image_size=192 --lr=1e-5 --dataset_path=celebA_100k --magnification_factor=4 --Degradation_type="DownBlur" --multiple_gpus="False" --Blur_radius=0.5 --vae_snapshot_name="VAE_finetuning_MSE_Perceptual" --diffusion_snapshot_name="Diffusion_finetuning" 
# python3 train_diffusion_superres.py --epochs=1 --noise_schedule="cosine" --batch_size=5 --image_size=192 --lr=1e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --patience=25  --dataset_path="celebA_10k" --inp_out_channels=3 --loss="MSE" --magnification_factor=4 --UNet_type="Residual Vision Multihead Attention UNet" --Degradation_type="DownBlur" --multiple_gpus="False" --ema_smoothing="True" --Blur_radius=0.5

python3 train_diffusion_superres.py --batch_size=4 --image_size=256 --noise_schedule="cosine" --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --patience=25  --dataset_path="up42_sentinel2_patches" --inp_out_channels=3 --loss="MSE" --magnification_factor=4 --UNet_type="Residual Attention UNet" --Degradation_type="DownBlur" --multiple_gpus="False" --ema_smoothing="False" --Blur_radius=0.5 --VAE_weight_path="$VAE_model_name"