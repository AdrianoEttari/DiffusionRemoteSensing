#!/bin/bash
model_name="TO_REMOVE_generate"

python3 train_diffusion_generation.py --epochs=1 --noise_schedule="cosine" --batch_size=5 --image_size=32 --lr=2e-4 --snapshot_name=snapshot.pt --model_name="$model_name" --noise_steps=2 --patience=25  --dataset_path="ImageNet256_small" --inp_out_channels=3 --loss="MSE" --UNet_type="DiffiT UNet" --multiple_gpus="False" --ema_smoothing="False" 

