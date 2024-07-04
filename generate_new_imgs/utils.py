import torch
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import cv2
 
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

    
if __name__=="__main__":
    pass