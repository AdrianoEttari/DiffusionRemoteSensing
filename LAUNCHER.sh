#!/bin/bash
model_name='TO_REMOVE'

#python3 train_diffusion_superres.py --epochs=1 --noise_schedule='cosine' --batch_size=5 --image_size=256 --lr=1e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --patience=25  --dataset_path='up42_sentinel2_patches' --inp_out_channels=3 --loss='MSE' --magnification_factor=2 --UNet_type='Residual Visual Multihead Attention UNet' --Degradation_type='DownBlur' --multiple_gpus='False' --ema_smoothing='True' --Blur_radius=0.5


python3 train_diffusion_superres.py --epochs=1 --noise_schedule='cosine' --batch_size=5 --image_size=192 --lr=1e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --patience=25  --dataset_path='celebA_10k' --inp_out_channels=3 --loss='MSE' --magnification_factor=4 --UNet_type='Residual Vision Multihead Attention UNet' --Degradation_type='DownBlur' --multiple_gpus='False' --ema_smoothing='True' --Blur_radius=0.5