import torch
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import cv2
from torch.optim.lr_scheduler import _LRScheduler
import math

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

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def get_real_to_model_classes_dict(num_classes):
    '''
                    TO ADJUST! IT MUST BE EMBEDDED INTO THE dataset BUILDER!

    PROBLEM: the classes are not sorted numerically, but lexicographically (i.e. '0','1','10','100','101','102',...,'99')
    So, when I ask the model to generate an image of the class 2, it will generate an image of the class 10.
    So, first of all I generate a dictionary (idx_to_class_model) that maps the class index to the class that the
    model actually generates. 
    Then, I generate a dictionary (real_to_model_classes_dict) that maps the class index to the class
    label that the model has given to the class index. This way, I can generate images of the classes that I want.

    Basically the returned dictionary says: "to get this class (index), you must give the model this class (value)"
    '''
    classes_list = [str(i) for i in range(num_classes)] 
    sorted_list = sorted(classes_list)
    idx_to_class_model = {idx:int(class_) for idx, class_ in enumerate(sorted_list)}
    values = list(idx_to_class_model.values())
    idxs = list(idx_to_class_model.keys())
    sort_mask = np.argsort(list(idx_to_class_model.values())) 
    sorted_idxs = [idxs[i] for i in sort_mask] # this is equal to sorted_folders

    real_to_model_classes_dict = {idxs[i]:int(sorted_idxs[i]) for i in range(len(sorted_idxs))} 
    return real_to_model_classes_dict

if __name__=="__main__":
    pass