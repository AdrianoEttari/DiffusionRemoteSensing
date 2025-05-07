#%%
import scipy.io
import os

file_path = os.path.join("..", "imagelabels.mat")
# Load the .mat file
data = scipy.io.loadmat(file_path)
# Access a specific variable
my_var = data['labels'][0]
print(my_var)

#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

dataset_path = os.path.join("..", "102flowers")
num_flowers = len(np.unique(my_var))

# %%
from tqdm import tqdm

images_names = os.listdir(dataset_path)

for i in tqdm(range(1, num_flowers+1)):
    flower_indices = np.argwhere(my_var == i).flatten()
    flower_images = [os.path.join(dataset_path, images_names[index]) for index in flower_indices]
    
    os.makedirs(os.path.join("..", "102flowers_dataset", str(i)), exist_ok=True)
    for j, flower_image in enumerate(flower_images):
        image = Image.open(flower_image).resize((512,512))
        image.save(os.path.join("..", "102flowers_dataset", str(i), f"{j}.jpg"))
        if j == 0:
            plt.imshow(image)
            plt.title(f"Flower {i}")
            plt.show()

# %%
