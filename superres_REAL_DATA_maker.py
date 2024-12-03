#%%
import numpy as np
from tqdm import tqdm
import rasterio
import os
from rasterio.windows import Window
from PIL import Image
#%%
img_folder = os.path.curdir

folder_output_path = os.path.join('landsat_sentinel_superres')
os.makedirs(os.path.join(folder_output_path, 'sentinel'), exist_ok=True)
os.makedirs(os.path.join(folder_output_path, 'landsat'), exist_ok=True)

sentinel_name_list = ['Brindisi_sentinel.tif', 'Sicilia_centro_sentinel.tif', 'Trento_Bolzano_sentinel.tif']
landsat_name_list = ['Brindisi_landsat.tif', 'Sicilia_centro_landsat.tif', 'Trento_Bolzano_landsat.tif']

band_name2index = {}


if len(landsat_name_list)>0:
    with rasterio.open(os.path.join(img_folder, landsat_name_list[0])) as src:
        num_bands = src.count
        for i in range(num_bands):
            band_desc = src.descriptions[i]
            band_name2index[band_desc] = i+1

if len(sentinel_name_list)>0:
    with rasterio.open(os.path.join(img_folder, sentinel_name_list[0])) as src:
        num_bands = src.count
        for i in range(num_bands):
            band_desc = src.descriptions[i]
            band_name2index[band_desc] = i+1

images_name_list = sentinel_name_list + landsat_name_list
print('Satellite images bands: ', band_name2index)
#%%
def process_image_in_chunks(img_path, chunk_size):
    with rasterio.open(img_path) as src:
        height, width = src.height, src.width
        profile = src.profile

        # Initialize an empty array for the full RGB image
        rgbnir_full = np.zeros((height, width, 4), dtype=np.uint8)

        # Iterate over chunks in both x and y dimensions
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                # Define window for the current chunk
                window = Window(x, y, min(chunk_size, width - x), min(chunk_size, height - y))

                # Read bands and process chunk
                if 'sentinel' in img_path:
                    rgbnir_chunk = src.read(
                        [band_name2index['B4'], band_name2index['B3'], band_name2index['B2'], band_name2index['B8']],
                        window=window
                    ) 
                elif 'landsat' in img_path:
                    rgbnir_chunk = src.read(
                        [band_name2index['SR_B4'], band_name2index['SR_B3'], band_name2index['SR_B2'], band_name2index['SR_B5']],
                        window=window
                    )

                # Transpose to (height, width, channels)
                rgbnir_chunk = rgbnir_chunk.transpose(1, 2, 0)

                # Normalize and convert to uint8
                rgbnir_min = np.nanmin(rgbnir_chunk)
                rgbnir_max = np.nanmax(rgbnir_chunk)
                rgbnir_normalized = (rgbnir_chunk - rgbnir_min) / (rgbnir_max - rgbnir_min)
                rgbnir_uint8_chunk = (rgbnir_normalized * 255).astype(np.uint8)

                # Place chunk into the final image array
                rgbnir_full[y:y + window.height, x:x + window.width, :] = rgbnir_uint8_chunk

    return rgbnir_full
#%%
images_name_list_filtered = images_name_list[:]
chunk_size = 5000

for img_name in tqdm(images_name_list_filtered):
    img_path = os.path.join(img_folder, img_name)
    # Process and construct the full image
    rgb_full = process_image_in_chunks(img_path, chunk_size)

    # Convert the final image array to a PIL Image and save as PNG in the correct folder
    image = Image.fromarray(rgb_full)
    img_name, ext = os.path.splitext(img_name)
    if 'sentinel' in img_name:
        image.save(os.path.join(folder_output_path, 'sentinel', img_name.split('_')[0]+'.png'))
    elif 'landsat' in img_name:
        image.save(os.path.join(folder_output_path, 'landsat', img_name.split('_')[0]+'.png'))

# %%
from PIL import Image
import numpy as np
import os
from Aggregation_Sampling import split_aggregation_sampling
from torchvision import transforms

sentinel_napoli = 'landsat_sentinel_superres/sentinel/Sfax.png'
transforms = transforms.ToTensor()
img_lr = transforms(np.array(Image.open(sentinel_napoli))).unsqueeze(0)

patch_size = 256
stride = 256
magnification_factor = 1
device = 'cpu'

patchifier = split_aggregation_sampling(img_lr, patch_size, stride, magnification_factor, device)
print(len(patchifier.patches_lr))
# %%
