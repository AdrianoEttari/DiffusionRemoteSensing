import numpy as np
from tqdm import tqdm
import rasterio
import os
from rasterio.windows import Window
from PIL import Image



img_folder = os.path.curdir
folder_output_path = os.path.curdir
images_name_list = ['caserta_sentinel.tif']
band_name2index = {}
with rasterio.open(os.path.join(img_folder, images_name_list[0])) as src:
    num_bands = src.count
    for i in range(num_bands):
        band_desc = src.descriptions[i]
        band_name2index[band_desc] = i+1
print('Satellite images bands: ', band_name2index)

def process_image_in_chunks(img_path, chunk_size):
    with rasterio.open(img_path) as src:
        height, width = src.height, src.width
        profile = src.profile

        # Initialize an empty array for the full RGB image
        rgb_full = np.zeros((height, width, 3), dtype=np.uint8)

        # Iterate over chunks in both x and y dimensions
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                # Define window for the current chunk
                window = Window(x, y, min(chunk_size, width - x), min(chunk_size, height - y))

                # Read bands and process chunk
                rgb_chunk = src.read(
                    [band_name2index['B4'], band_name2index['B3'], band_name2index['B2']],
                    window=window
                ) 

                # Transpose to (height, width, channels)
                rgb_chunk = rgb_chunk.transpose(1, 2, 0)

                # Normalize and convert to uint8
                rgb_min = np.nanmin(rgb_chunk)
                rgb_max = np.nanmax(rgb_chunk)
                rgb_normalized = (rgb_chunk - rgb_min) / (rgb_max - rgb_min)
                rgb_uint8_chunk = (rgb_normalized * 255).astype(np.uint8)

                # Place chunk into the final image array
                rgb_full[y:y + window.height, x:x + window.width, :] = rgb_uint8_chunk

    return rgb_full

images_name_list_filtered = images_name_list[:]
chunk_size = 5000

for img_name in tqdm(images_name_list_filtered):
    img_path = os.path.join(img_folder, img_name)
    # Process and construct the full image
    rgb_full = process_image_in_chunks(img_path, chunk_size)

    # Convert the final image array to a PIL Image and save as PNG
    image = Image.fromarray(rgb_full)
    img_name, ext = os.path.splitext(img_name)
    img_name_2save = str(img_name)+'.png'
    image.save(os.path.join(folder_output_path, img_name_2save))