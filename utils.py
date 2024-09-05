from torch.utils.data import Dataset
import torch
import numpy as np
import os
from torchvision import transforms
import shutil
import random
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from scipy.linalg import orth
from degradation_from_BSRGAN import degradation_bsrgan_plus, single2uint, imread_uint, soft_degradation_bsrgan
import imageio
import cv2
import torch.nn.functional as F

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    '''
    This function adds Gaussian noise to the image. The noise level is randomly chosen between noise_level1 and noise_level2.
    There are three possibilities of adding noise depending on the value of rnum:
    - rnum > 0.6: add color Gaussian noise (add to the image a sample from a normal distribution with mean=0 and std=noise_level/255.0 of the same shape of the image).
    - rnum < 0.4: add grayscale Gaussian noise (add to the image a sample from a normal distribution with mean=0 and std=noise_level/255.0 of shape (height_img,width_img,1)).
    - 0.4 < rnum < 0.6: add noise (add to the image a sample from a multivariate normal distribution with mean=[0,0,0] and covariance matrix=abs(L**2*conv) of shape (height_img,width_img,3), where L is a random number between 0 and 1 and conv is a random covariance matrix).
    '''
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    img = img.permute(1,2,0).numpy()
    if rnum > 0.6:   # add color Gaussian noise (add to the image a sample from a normal distribution with mean=0 and std=noise_level/255.0 of the same shape of the image)
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise (add to the image a sample from a normal distribution with mean=0 and std=noise_level/255.0 of shape (height_img,width_img,1))
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255. # noise_level2 is assumed to be in the range [0,255] so, let's bring to the range [0,1]
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3)) # computes the orthogonal basis for the range of a matrix.
        conv = np.dot(np.dot(np.transpose(U), D), U) # The covariance matrix (conv) is constructed by transforming the diagonal matrix (D) with the orthogonal matrix (U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    img = torch.tensor(img).permute(2,0,1).to(torch.float)
    return img

class get_data_SAR_TO_NDVI(Dataset):
    '''
    This class allows to store the data in a Dataset that can be used in a DataLoader
    like that train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True).

    -Input:
        root_dir: path to the folder where the data is stored. 
        transform: a torchvision.transforms.Compose object with the transformations that will be applied to the images.
        data_format: 'PIL' or 'numpy' or 'torch'. The format of the images in the dataset.
    -Output:
        A Dataset object that can be used in a DataLoader.

    __getitem__ returns sar_img and ndvi_img. The split in batches must be done in the DataLoader (not here).
    '''
    def __init__(self, root_dir, transform=None, data_format='torch'):
        self.root_dir = root_dir
        self.transform = transform
        self.opt_path = os.path.join(self.root_dir, 'opt')
        self.sar_path = os.path.join(self.root_dir, 'sar')
        self.sar_ndvi_filenames = sorted(os.listdir(self.sar_path))
        self.data_format = data_format
    
    def __len__(self):
        return len(self.sar_ndvi_filenames)

    def __getitem__(self, idx):
        sar_path = os.path.join(self.sar_path, self.sar_ndvi_filenames[idx])
        ndvi_path = os.path.join(self.opt_path, self.sar_ndvi_filenames[idx])

        if self.data_format == 'PIL':
            sar_img = Image.open(sar_path)
            ndvi_img = Image.open(ndvi_path)
            sar_img = transforms.ToTensor()(sar_img)
            ndvi_img = transforms.ToTensor()(ndvi_img)
        elif self.data_format == 'numpy':
            sar_img = np.load(sar_path)
            ndvi_img = np.load(ndvi_path)
            sar_img = torch.tensor(sar_img).to(torch.float)
            ndvi_img = torch.tensor(ndvi_img).to(torch.float)
        elif self.data_format == 'torch':
            sar_img = torch.load(sar_path)
            ndvi_img = torch.load(ndvi_path)

        if self.transform:
            sar_img = self.transform(sar_img)
            ndvi_img = self.transform(ndvi_img)

        # Bring the images to the range [0,1] (assume they are in the range [-1,1])
        sar_img = (sar_img+1)/2
        ndvi_img = (ndvi_img+1)/2
        
        return sar_img, ndvi_img
    
class get_data_superres(Dataset):
    '''
    This class allows to store the data in a Dataset that can be used in a DataLoader
    like that train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True).
    First a downsample is applied to the original images, then a Gaussian blur is applied to the downsampled images (the user may choose between
    a random blur or a fixed one) and finally there is the option to add Gaussian noise to the downsampled images. The __getitem__
    function returns the downsampled image (x) and the original image (y).

    -Input:
        root_dir: path to the folder where the data is stored. 
        magnification_factor: factor by which the original images are downsampled.
        blur_radius: radius of the Gaussian blur that will be applied to the downsampled images.
        Gauss_noise: boolean. If True, Gaussian noise will be added to the downsampled images.
        data_format: 'PIL' or 'numpy' or 'torch'. The format of the images in the dataset.
        transform: a torchvision.transforms.Compose object with the transformations that will be applied to the images.
    -Output:
        A Dataset object that can be used in a DataLoader.

    __getitem__ returns x and y. The split in batches must be done in the DataLoader (not here).
    '''
    def __init__(self, root_dir, magnification_factor, blur_radius=0.5, Gauss_noise=False, data_format='PIL', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.magnification_factor = magnification_factor    
        self.original_imgs_dir = os.path.join(self.root_dir)
        self.blur_radius = blur_radius
        self.Gauss_noise = Gauss_noise
        self.data_format = data_format
        self.y_filenames = sorted(os.listdir(self.original_imgs_dir))
    
    def __len__(self):
        return len(self.y_filenames)

    def __getitem__(self, idx):
        y_path = os.path.join(self.original_imgs_dir, self.y_filenames[idx])

        if self.data_format == 'PIL':
            y = Image.open(y_path)
        elif self.data_format == 'numpy':
            y = np.load(y_path)
            y = Image.fromarray((y*255).astype(np.uint8))
        elif self.data_format == 'torch':
            to_pil = transforms.ToPILImage()
            y = torch.load(y_path)
            y = to_pil(y)

        if self.transform:
            y = self.transform(y)

        # Downsample the original image
        downsample = transforms.Resize((y.size[0] // self.magnification_factor, y.size[1] // self.magnification_factor),
                                       interpolation=transforms.InterpolationMode.BICUBIC)
        try:
            x = downsample(y)
        except:
            x = downsample(y.to('cpu')).to(y.device)
    
        # Add blur
        if self.blur_radius == 'random':
            self.blur_radius = random.triangular(0.5,1.5,1)

        x = x.filter(ImageFilter.GaussianBlur(self.blur_radius))

        to_tensor = transforms.ToTensor()
        x = to_tensor(x)
        y = to_tensor(y)

        # Add Gaussian noise
        if self.Gauss_noise == False:
            x = x
        elif self.Gauss_noise == True:
            x = add_Gaussian_noise(x, noise_level1=2, noise_level2=10)
            
        return x, y
        
class get_data_superres_BSRGAN(Dataset):
    '''
    This class allows to store the data in a Dataset that can be used in a DataLoader
    like that train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True).

    -Input:
        root_dir: path to the folder where the data is stored. 
        magnification_factor: factor by which the original images are downsampled.
        model_input_size: size of the input images to the model.
        num_crops: number of crops to be generated from each image.
        degradation_type: 'BSR_plus' or 'soft_BSR_plus'. The type of degradation that will be applied to the images.
        destination_folder: path to the folder where the lr and hr images will be saved.
    -Output:
        A Dataset object that can be used in a DataLoader.

    __getitem__ returns x and y. The split in batches must be done in the DataLoader (not here).
    '''
    def __init__(self, root_dir, magnification_factor, model_input_size, num_crops, degradation_type='BSR_plus', destination_folder=None):
        self.root_dir = root_dir
        self.magnification_factor = magnification_factor
        self.model_input_size = model_input_size
        self.original_imgs_dir = os.path.join(self.root_dir)
        self.y_filenames = sorted(os.listdir(self.original_imgs_dir))
        self.num_crops = num_crops
        self.degradation_type = degradation_type
        self.x_images, self.y_images = self.BSR_degradation()
        if destination_folder is not None:
            self.dataset_saver(destination_folder)

    def BSR_degradation(self):
        '''
        This function takes as input the path of the original images, the magnification factor, the model input size
        and also the the number of crops to be generated from each image.
        It returns two lists, one with the lr images and another with the hr images.
        '''
        x_images = []
        y_images = []
        for i in tqdm(range(len(self.y_filenames))):
            y_path = os.path.join(self.original_imgs_dir, self.y_filenames[i])
            for _ in range(self.num_crops):
                y = imread_uint(y_path, 3)
                if self.degradation_type == 'BSR_plus':
                    x, y = degradation_bsrgan_plus(y, sf=self.magnification_factor, lq_patchsize=self.model_input_size)
                elif self.degradation_type == 'soft_BSR_plus':
                    x, y = soft_degradation_bsrgan(y, sf=self.magnification_factor, lq_patchsize=self.model_input_size)
                x = single2uint(x)
                y = single2uint(y)
                to_tensor = transforms.ToTensor()
                x = to_tensor(x)
                y = to_tensor(y)
                x_images.append(x)
                y_images.append(y)
        
        # Shuffle the lists
        combined = list(zip(x_images, y_images))
        random.shuffle(combined)
        x_images[:], y_images[:] = zip(*combined)

        return x_images, y_images

    def dataset_saver(self, destination_folder):
        '''
        This function saves the lr and hr images in the destination_folder with the following paths: 
        <destination_folder>/lr/x_<i>.png and <destination_folder>/hr/y_<i>.png.
        '''
        os.makedirs(destination_folder, exist_ok=True)
        os.makedirs(os.path.join(destination_folder, 'lr'), exist_ok=True)
        os.makedirs(os.path.join(destination_folder, 'hr'), exist_ok=True)
        for i in range(len(self.x_images)):
            x = self.x_images[i]
            y = self.y_images[i]
            x_path = os.path.join(destination_folder, 'lr',  f"x_{i}.png")
            y_path = os.path.join(destination_folder, 'hr',  f"y_{i}.png")
            x = x.permute(1, 2, 0).clamp(0, 1).numpy()
            y = y.permute(1, 2, 0).clamp(0, 1).numpy()
            x = Image.fromarray((x * 255).astype(np.uint8))
            y = Image.fromarray((y * 255).astype(np.uint8))
            x.save(x_path)
            y.save(y_path)

    def __len__(self):
        return len(self.x_images)

    def __getitem__(self, idx):
        x = self.x_images[idx]
        y = self.y_images[idx]

        return x, y
    
class data_organizer_superresolution():
    '''
    This class allows to organize the data inside main_folder (provided in the __init__) 
    into train_original, val_original and test_original folders that will be created inside
    main_folder.
    ATTENTION: it is tailored for the super resolution problem.
    '''
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.train_folder = os.path.join(main_folder, 'train_original')
        self.val_folder = os.path.join(main_folder, 'val_original')
        self.test_folder = os.path.join(main_folder, 'test_original')
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.val_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
            
    def get_all_files_in_folder_and_subfolders(self, folder):
        '''
        Input:
            folder: path to the folder where the files and subfolders are stored.

        Output:
            all_files: list with the full path of all the files in the folder and its subfolders.
        '''
        all_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        return all_files

    def split_files(self, split_ratio=(0.8, 0.15, 0.05)):
        '''
        This function splits the files in the main folder into train, val and test folders.

        Input:
            split_ratio: tuple with the ratio of files that will be assigned to the train, val and test folders.
        Output:
            None
        '''
        all_files = self.get_all_files_in_folder_and_subfolders(self.main_folder)
        # Get a list of all files in the input folder
        random.shuffle(all_files)  # Shuffle the files randomly

        # Calculate the number of files for each split
        total_files = len(all_files)
        train_size = int(total_files * split_ratio[0])
        val_size = int(total_files * split_ratio[1])

        # Assign files to the respective splits
        train_files = all_files[:train_size]
        val_files = all_files[train_size:train_size + val_size]
        test_files = all_files[train_size + val_size:]

        # Move files to the output folders
        self.move_files(train_files, self.train_folder)
        self.move_files(val_files, self.val_folder)
        self.move_files(test_files, self.test_folder)

    def move_files(self, files_full_path, destination_folder):
        '''
        This function moves the files to the destination folder.

        Input:
            files_full_path: list with the full path of the files that will be moved.
            destination_folder: path to the folder where the files will be moved.

        Output:
            None
        '''
        for file_full_path in tqdm(files_full_path,desc='Moving files'):
            destination_path = os.path.join(destination_folder, os.path.basename(file_full_path))
            shutil.move(file_full_path, destination_path)

def convert_png_to_jpg(png_file, jpg_file):
    try:
        # Open the PNG image
        with Image.open(png_file) as img:
            # Convert RGBA images to RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # Save as JPG
            img.save(jpg_file, 'JPEG')
        print("Conversion successful!")
    except Exception as e:
        print("Conversion failed:", e)

def gif_maker(frames, frame_stride=1, destination_path='output.gif'):
    '''
    This function saves the frames that are passed in input as a gif.

    *Input:
        - frames: list of frames to be saved in a gif
        - frame_stride: int. The jump between frames that will be saved in the gif (e.g. if frame_stride=5 i=0 frame is considered, then i=5 frame is considered, etc.)
    *Output:
        - None
    '''
    images = []
    if frames[0].max()<100:
        frames = [torch.clamp(frame[0],0,1)*255 for frame in frames]
    
    for i,frame in enumerate(tqdm(frames)):
        if (i % frame_stride == 0) or (i == len(frames)-1):
            frame = frame.type(torch.uint8)
            np_frame = frame.permute(1, 2, 0).detach().cpu().numpy()
            image = Image.fromarray(np_frame)

            draw = ImageDraw.Draw(image)

            font = ImageFont.load_default()
            text = f'frame {i}'

            text_position = (10, 10)
            text_color = (255, 255, 255)  # white text
            
            outline_color = (0, 0, 0)  # black outline
            draw.text((text_position[0]-1, text_position[1]-1), text, font=font, fill=outline_color)
            draw.text((text_position[0]+1, text_position[1]-1), text, font=font, fill=outline_color)
            draw.text((text_position[0]-1, text_position[1]+1), text, font=font, fill=outline_color)
            draw.text((text_position[0]+1, text_position[1]+1), text, font=font, fill=outline_color)

            draw.text(text_position, text, font=font, fill=text_color)

            images.append(image)

    imageio.mimsave(destination_path, images, duration=0.0005)

def video_maker(frames, video_path='output.mp4', fps=50):
    '''
    Convert a sequence of frames to a video.

    *Input:
        - frames: list of frames to be saved in a video
        - video_path: path to save the video file
        - fps: frames per second
    *Output:
        - None
    '''
    if frames[0].max() < 100:
        frames = [torch.clamp(frame[0], 0, 1) * 255 for frame in frames]
    
    height, width = frames[0][0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    print('Creating video... with frames:', len(frames))
    for i,frame in enumerate(frames):
        frame = frame.type(torch.uint8)
        np_frame = frame.permute(1, 2, 0).detach().cpu().numpy()
        frame_bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)

        # Create an Image object for drawing
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        # Add text (frame title)
        font = ImageFont.load_default()

        text = f'Frame {i}'

        text_position = (10, 10)
        text_color = (255, 255, 255)  # white text
        outline_color = (0, 0, 0)  # black outline
        draw.text((text_position[0] - 1, text_position[1] - 1), text, font=font, fill=outline_color)
        draw.text((text_position[0] + 1, text_position[1] - 1), text, font=font, fill=outline_color)
        draw.text((text_position[0] - 1, text_position[1] + 1), text, font=font, fill=outline_color)
        draw.text((text_position[0] + 1, text_position[1] + 1), text, font=font, fill=outline_color)
        draw.text(text_position, text, font=font, fill=text_color)

        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Write frame to video
        video.write(frame_bgr)
    
    video.release()
    # to convert an mp4 into gif, run into terminal ffmpeg -i <input.mp4> -vf "fps=100,scale=320:-1:flags=lanczos" <output.gif>

def calculate_entropy(bottleneck):
    # Flatten the bottleneck (batch_size, channels, height, width) -> (batch_size, -1)
    batch_size = bottleneck.size(0)
    bottleneck_flat = bottleneck.view(batch_size, -1)

    # Normalize to get a probability distribution (softmax or simple normalization)
    probabilities = F.softmax(bottleneck_flat, dim=1)
    
    # Calculate the Shannon entropy for each sample in the batch
    entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=1)
    
    # Return the average entropy across the batch
    return torch.mean(entropy)
    
if __name__=="__main__":
    main_folder = 'up42_sentinel2_patches'
    data_organizer = data_organizer_superresolution(main_folder)
    data_organizer.split_files(split_ratio=(0.85,0.1,0.05))
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))

    # source_folder = 'satellite_imgs_test'
    # destination_folder = 'satellite_imgs_test_cropped'
    # desired_width = 128
    # img_splitter(source_folder, destination_folder, desired_width)
