# Adaptation of Diffusion Models for Remote Sensing Imagery

## Summary of the Files and Folders 

- **train_diffusion_superres.py**: Contains the diffusion model class with all necessary functions for sampling and training.

- **UNet_model_superres.py**: Defines the UNet model used by the diffusion model to denoise images for the super-resolution task.

- **train_diffusion_SAR_TO_NDVI.py** and **UNet_model_SAR_TO_NDVI.py**: Used for solving the SAR to NDVI problem, structured similarly to the super-resolution models but using SAR images as inputs and NDVI images as outputs.

- **superres_and_NDVIgen.py**: Contains functions for generating and plotting both super-resolution images and SAR-to-NDVI converted images.

- **utils.py**: Includes functions to create the data folder for the super-resolution task (data_organizer()) and the get_data functions for both super-resolution and SAR to NDVI tasks, which prepare the dataset for the dataloader.

- **models_representation.pptx**: A PowerPoint file with graphical representations of the UNet model architectures.

- **Aggregation_Sampling.py**: Used at inference time to split an image into multiple patches, super resolve each patch, and then reassemble them, following the aggregation sampling method described in [this paper](https://arxiv.org/abs/2305.07015).

- **qr_code_builder.py**: Generates a QR code based on the provided URL.

### Generative Part

- **train_diffusion_generation.py** and **UNet_model_generation.py**: Training and UNet model files for generating new images, located in the **generate_new_imgs** folder.

- **utils.py**: (it is in the generate_new_imgs folder) Similar to the one in the main folder but without data folder and dataset creation functions. In this case, the data folder must contain a separate folder for each class, with images inside each class folder.

- **imgs_generator.py**: Equivalent to superres_and_NDVIgen.py but for the generative case.

### Models Run

- **models_run**: Contains different models with their weights and some results relative to the datasets they were trained on.


🚧🚧🚧🚧🚧  WORK IN PROGRESS 🚧🚧🚧🚧🚧

In **degradation_from_BSRGAN.py** there are functions taken from https://github.com/IceClear/StableSR to degrade the images in a more realistic way; these functions are then applied in the function get_data_superres_BSRGAN() of utils.py.  
<!-- In the folder **multihead_attention** there are files to implement the multihead attention mechanism in the UNet model instead of the simple attention. -->

## RESULTS

### SUPER RESOLUTION

[<img src="assets/imgsli_up42.png" height="400px"/>](https://imgsli.com/Mjc2NjAw)

Here is a video showcasing the denoising process for the super resolution problem of the Sentinel-2 dataset:

![Video Denoising](https://github.com/AdrianoEttari/DiffusionRemoteSensing/blob/main/assets/UP42_SUPERRESOLUTION/DownBlur/up42_superresolution.gif)

### IMAGE GENERATION of https://github.com/phelber/EuroSAT. 
The first set of images is the real one and the second set is the generated one.

<img src="assets/EuroSat_real.png" height="250px"/>
<img src="assets/EuroSat_predictions.png" height="265px"/>

### SAR TO NDVI

NDVI G.T. stands for Ground Truth NDVI, while NDVI PRED. stands for the predicted NDVI from the SAR image.

<img src="assets/SAR_to_NDVI.png" height="400px"/>

## TODO
- [ ] Add MultiHead Attention from Vision Transformer 
- [ ] Incorporate the Diffusion Model in a Latent Diffusion Model
- [ ] Substitute the simple Blur-Down-Gauss degradation with the BSR-degradation algorithm
- [x] ~~Add Aggregation Sampling~~

## Train (snippet to train a super resolution model on the UP42 dataset with the DownBlur degradation)
```
python3 train_diffusion_superres.py --epochs=1001 --noise_schedule='cosine' --batch_size=16 --image_size=256 --lr=1e-4 --snapshot_name=snapshot.pt --model_name="Residual_Attention_UNet_superres_magnification2_LRimgsize128_up42_sentinel2_patches_downblur" --noise_steps=1500 --patience=50  --dataset_path='up42_sentinel2_patches' --inp_out_channels=3 --loss='MSE' --magnification_factor=2 --UNet_type='Residual Attention UNet' --Degradation_type='DownBlur' --multiple_gpus='False' --ema_smoothing='True' --Blur_radius=0.5
```
## Contact
If you have any questions, feel free to contact me at `adriano.ettari@unina.it` or on my LinkedIn page [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adriano-ettari-b8741b21b/)

