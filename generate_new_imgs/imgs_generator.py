# %%
from train_diffusion_generation import Diffusion
from UNet_model_generation_VMHA import Residual_Attention_UNet_generation, Residual_DiffiT_UNet_generation
import torch
import matplotlib.pyplot as plt
import os
from utils import get_real_to_model_classes_dict

noise_schedule = 'cosine'
input_channels = output_channels = 3
device = 'mps'
noise_steps = 1500
# model_name = 'DiffiT_UNet_generation_TinyImageNet200'
model_name = 'Residual_Attention_UNet_generation_TinyImageNet200'
snapshot_path = os.path.join('..', 'models_run', model_name, 'weights', 'snapshot.pt')

image_size = 64
num_classes = 200
model = Residual_Attention_UNet_generation(input_channels, output_channels, num_classes, device).to(device)
# model = Residual_DiffiT_UNet_generation(input_channels, output_channels, num_classes, device).to(device)

diffusion = Diffusion(
        noise_schedule=noise_schedule, model=model,
        snapshot_path=snapshot_path,
        noise_steps=noise_steps, beta_start=1e-4, beta_end=0.02,
        device=device, image_size=image_size, model_name=model_name,
        multiple_gpus=False, ema_smoothing=False)

######### Tiny ImageNet 200 ############
classes = [0,1,2,3,4,5,6,7,8,9]


########### Sentinel Data Crops ############
# classes = ['Highway', 'River', 'HerbaceousVegetation','Residential', 'AnnualCrop',
#             'Pasture', 'Forest', 'PermanentCrop', 'Industrial', 'SeaLake']

############ CIFAR10 ############
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

save_path = os.path.join('..', 'models_run', model_name, 'results', 'generated_imgs')
classes = sorted(classes)
fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # Adjust figsize as needed
axs = axs.ravel()

real_to_model_classes_dict = get_real_to_model_classes_dict(num_classes)

for i, class_ in enumerate(classes):
        class_model = real_to_model_classes_dict[class_]
        prediction = diffusion.sample(n=1,model=model, target_class=torch.tensor([class_model], dtype=torch.int64).to(device), input_channels=input_channels, generate_video=False)
        prediction = prediction.clamp(0, 1)
        axs[i].imshow(prediction[0].permute(1,2,0).detach().cpu())
        axs[i].axis('off')
        axs[i].set_title(class_, fontsize=12)
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
# %%
