# %%
from generate_new_imgs.train_diffusion_generation import Diffusion
from UNet_model_generation import Residual_Attention_UNet_generation
import torch
import matplotlib.pyplot as plt
import os

noise_schedule = 'cosine'
input_channels = output_channels = 3
device = 'cpu'
noise_steps = 1500
model_name = 'Residual_Attention_UNet_generation_sentinel_data_crops'
snapshot_path = os.path.join('..', 'models_run', model_name, 'weights', 'snapshot.pt')

image_size = 64

model = Residual_Attention_UNet_generation(input_channels, output_channels, 10, device).to(device)

diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02,
        device=device, image_size=image_size, model_name=model_name,
        multiple_gpus=False, ema_smoothing=False)


########### Sentinel Data Crops ############
classes = ['Highway', 'River', 'HerbaceousVegetation','Residential', 'AnnualCrop',
            'Pasture', 'Forest', 'PermanentCrop', 'Industrial', 'SeaLake']

############ CIFAR10 ############
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

classes = sorted(classes)
fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # Adjust figsize as needed
axs = axs.ravel()
for i, class_ in enumerate(classes):
        prediction = diffusion.sample(n=1,model=model, target_class=torch.tensor([i], dtype=torch.int64).to(device), input_channels=input_channels, generate_video=False)
        prediction = prediction.clamp(0, 1)
        axs[i].imshow(prediction[0].permute(1,2,0).detach().cpu())
        axs[i].axis('off')
        axs[i].set_title(class_, fontsize=12)
plt.show()
# %%
