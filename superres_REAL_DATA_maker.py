#%% IMPORT LIBRARIES
import numpy as np
from tqdm import tqdm
import rasterio
import os
from rasterio.windows import Window
from PIL import Image

#%% BUILD A DICTIONARY WITH THE BANDS NAMES AND THEIR INDEXES. AND BUILD A LIST WITH ALL THE IMAGES NAMES
img_folder = os.path.curdir

folder_output_path = os.path.join('landsat_sentinel_superres')
os.makedirs(os.path.join(folder_output_path, 'sentinel'), exist_ok=True)
os.makedirs(os.path.join(folder_output_path, 'landsat'), exist_ok=True)

sentinel_name_list = ['Napoli_sentinel.tif']
landsat_name_list = ['Napoli_landsat.tif']

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

#%% PREPROCESSING OF THE IMAGES
def SCL_mask_maker(tif_image_path, band_name2index):
    tif_image = rasterio.open(tif_image_path)
    SCL = tif_image.read(band_name2index['SCL'])
    SCL = np.round(SCL * 10000).astype(int)
    SCL = SCL.astype(np.uint16)
    not_valid_classes = [0,1,3,6,7,8,9,10]
    mask = np.isin(SCL, not_valid_classes)
    return mask

def landsat_mask_maker(tif_image_path):
    tif_image = rasterio.open(tif_image_path)
    tif_image = tif_image.read(1)
    mask = np.isnan(tif_image)
    return mask
#%% FUNCTION TO PROCESS THE IMAGES IN CHUNKS

def process_image_in_chunks(img_path, chunk_size, band_name2index):
    with rasterio.open(img_path) as src:
        height, width = src.height, src.width
        profile = src.profile
        
        if 'sentinel' in img_path:
            scl_mask = SCL_mask_maker(img_path, band_name2index)
            landsat_mask = None
        elif 'landsat' in img_path:
            landsat_mask = landsat_mask_maker(img_path)
            scl_mask = None
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

    return rgbnir_full, scl_mask, landsat_mask
#%% EXECUTE THE FUNCTION FOR EACH IMAGE AND SAVE THE FINAL IMAGE
images_name_list_filtered = images_name_list[:]
chunk_size = 5000
scl_masks = {}
landsat_masks = {}
for img_name in tqdm(images_name_list_filtered):
    img_path = os.path.join(img_folder, img_name)
    # Process and construct the full image
    rgbnir_full, scl_mask, landsat_mask = process_image_in_chunks(img_path, chunk_size, band_name2index)
    if scl_mask is not None:
        scl_masks[img_name] = scl_mask
    if landsat_mask is not None:
        landsat_masks[img_name] = landsat_mask
    # Convert the final image array to a PIL Image and save as PNG in the correct folder
    image = Image.fromarray(rgbnir_full)
    img_name, ext = os.path.splitext(img_name)
    if 'sentinel' in img_name:
        image.save(os.path.join(folder_output_path, 'sentinel', img_name.split('_')[0]+'.png'))
    elif 'landsat' in img_name:
        image.save(os.path.join(folder_output_path, 'landsat', img_name.split('_')[0]+'.png'))

# %% PATCHIFICATION OF THE IMAGES
from PIL import Image
import numpy as np
import os
from Aggregation_Sampling import split_aggregation_sampling
from torchvision import transforms
from tqdm import tqdm
import cv2

full_imgs_folder_path = 'landsat_sentinel_superres'
patches_folder_path = 'landsat_sentinel_superres_patches'
os.makedirs(os.path.join(patches_folder_path, 'landsat'), exist_ok=True)
os.makedirs(os.path.join(patches_folder_path, 'sentinel'), exist_ok=True)

discarded = 0
for img_name in tqdm(os.listdir(os.path.join(full_imgs_folder_path, 'sentinel'))):
    landsat_img_path = os.path.join(full_imgs_folder_path, 'landsat', img_name)
    sentinel_img_path = os.path.join(full_imgs_folder_path, 'sentinel', img_name)
    landsat_img = Image.open(landsat_img_path)
    sentinel_img = Image.open(sentinel_img_path)

    if sentinel_img.size[0] > landsat_img.size[0]*2 and sentinel_img.size[0] < landsat_img.size[0]*4 and sentinel_img.size[1] > landsat_img.size[1]*2 and sentinel_img.size[1] < landsat_img.size[1]*4:
        landsat_img = landsat_img.resize((sentinel_img.size[0], sentinel_img.size[1]), Image.BICUBIC)
    else:
        raise ValueError('The size of the images is not correct')

    transform = transforms.ToTensor()

    sentinel_img = np.array(sentinel_img)
    scl_mask = scl_masks[img_name.replace('.png', '_sentinel.tif')][..., None]
    sentinel_img = np.concatenate((sentinel_img, scl_mask), axis=2)
    sentinel_img = transform(sentinel_img).unsqueeze(0)
    
    landsat_img = np.array(landsat_img)
    landsat_mask = landsat_masks[img_name.replace('.png', '_landsat.tif')][..., None]
    target_size = (landsat_img.shape[1], landsat_img.shape[0])
    landsat_mask = landsat_mask.astype(np.uint8)
    landsat_mask = cv2.resize(landsat_mask, target_size, interpolation=cv2.INTER_NEAREST)[..., None]
    landsat_img = np.concatenate((landsat_img, landsat_mask), axis=2)
    landsat_img = transform(landsat_img).unsqueeze(0)

    patch_size = 256
    stride = 256
    magnification_factor = 1
    device = 'cpu'
    patchifier_landsat = split_aggregation_sampling(landsat_img, patch_size, stride, magnification_factor, device)
    patchifier_sentinel = split_aggregation_sampling(sentinel_img, patch_size, stride, magnification_factor, device)

    for i in range(len(patchifier_sentinel.patches_lr)):
        patch_landsat = patchifier_landsat.patches_lr[i].squeeze(0)
        patch_sentinel = patchifier_sentinel.patches_lr[i].squeeze(0)
        scl_patch_mask = patchifier_sentinel.patches_lr[i][0].permute(1,2,0)[:,:,4]
        landsat_patch_mask = patchifier_landsat.patches_lr[i][0].permute(1,2,0)[:,:,4]

        if scl_patch_mask.sum() == 0 and landsat_patch_mask.sum() == 0:
            patch_landsat = Image.fromarray((patch_landsat.permute(1,2,0)[:,:,:3].cpu().numpy()*255).astype(np.uint8))
            patch_landsat.save(os.path.join(patches_folder_path, 'landsat', img_name.split('.')[0] + '_patch_' + str(i) + '.png'))
            patch_sentinel = Image.fromarray((patch_sentinel.permute(1,2,0)[:,:,:3].cpu().numpy()*255).astype(np.uint8))
            patch_sentinel.save(os.path.join(patches_folder_path, 'sentinel', img_name.split('.')[0] + '_patch_' + str(i) + '.png'))
        else:
            discarded+=1

        


# %% TO REMOVE (BUILD DATASET AND CHECK ONE IMAGE BAND)
from utils import get_data_superres_REAL_DATA
import matplotlib.pyplot as plt

dataset = get_data_superres_REAL_DATA('landsat_sentinel_superres_patches')

for i in range(10):
    fig, axs = plt.subplots(1,2)
    axs = axs.ravel()
    axs[0].imshow(dataset[i][0].permute(1,2,0)[:,:,0], cmap='gray')
    axs[0].set_title('Landsat')
    axs[1].imshow(dataset[i][1].permute(1,2,0)[:,:,0], cmap='gray')
    axs[1].set_title('Sentinel')
    plt.show()

# %% TO REMOVE (CHECK ALL THE BANDS OF BOTH LANDSAT AND SENTINEL FOR EACH IMAGE)
transform = transforms.ToTensor()

for patch in os.listdir(os.path.join(patches_folder_path, 'landsat')):
    fig, axs = plt.subplots(2,4)

    landsat_img = np.array(Image.open(os.path.join(patches_folder_path, 'landsat', patch)))
    landsat_img = transform(landsat_img)

    sentinel_img = np.array(Image.open(os.path.join(patches_folder_path, 'sentinel', patch)))
    sentinel_img = transform(sentinel_img)

    print(patch)
    for j in range(4):
        axs[0,j].imshow(landsat_img.permute(1,2,0)[:,:,j], cmap='gray')
        axs[0,j].set_title('Landsat band ' + str(j+1))
        axs[0,j].axis('off')
        axs[1,j].imshow(sentinel_img.permute(1,2,0)[:,:,j], cmap='gray')
        axs[1,j].set_title('Sentinel band ' + str(j+1))
        axs[1,j].axis('off')
    plt.show()

    _input = input('Press c to continue')
    if _input == 'c':
        continue
    else:
        break

# %%
