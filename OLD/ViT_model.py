import timm
import torch.nn as nn
from torchvision import transforms
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class RRDB(nn.Module):
    '''
    This class is used in order to encode the low-resolution image.
    It is composed of a series of residual blocks.
    '''
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super(RRDB, self).__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(in_channels, in_channels) for _ in range(num_blocks)]
        )
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.blocks(x)
        out = self.conv_out(out)
        out += x
        return out
    
# Define the model class
class ViTModel(nn.Module):
    def __init__(self, image_channels, pretrained_model_name='vit_base_patch16_224', device='cuda'):
        super(ViTModel, self).__init__()
        self.device = device
        self.image_channels = image_channels
        self.time_emb_dim = 100
        self.conv0 = nn.Conv2d(self.image_channels, self.image_channels, 3, padding=1)
        self.vit = timm.create_model(pretrained_model_name, pretrained=True)
        self.LR_encoder = RRDB(in_channels=self.image_channels, out_channels=self.image_channels, num_blocks=3)
        self.conv_upsampled_lr_img = nn.Conv2d(self.image_channels, self.image_channels, 3, padding=1)
        # Adapt the classifier to return an image of the same shape as input (3, 256, 256)
        self.vit.head = nn.Sequential(
            nn.Linear(self.vit.head.in_features, 256 * 256 * 3),
            nn.Unflatten(1, (3, 256, 256))
        )
        self.relu = nn.ReLU(inplace=False)
        # Add additional layers if needed for more processing
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.time_mlp =  self._make_te(self.time_emb_dim, image_channels, device=device)
        self.batch_norm = nn.BatchNorm2d(image_channels)

    def pos_encoding(self, t, channels, device):
        inv_freq = 1.0 / (
            10000**(torch.arange(0, channels, 2, device= device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def _make_te(self, dim_in, dim_out, device):
        '''
        This function creates a time embedding layer.
        '''
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out, device=device)
        )
    
    def forward(self,  x, timestep, lr_img, magnification_factor):
        image_size = x.shape[-1]

        t = timestep.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim, device=self.device)
        t = self.relu(self.time_mlp(t))
        t = t[(..., ) + (None, ) * 2]

        lr_img = self.LR_encoder(lr_img)
        # UPSAMPLE LR IMAGE
        try:
            upsampled_lr_img = F.interpolate(lr_img, scale_factor=magnification_factor, mode='bicubic')
        except:
            upsampled_lr_img = F.interpolate(lr_img.to('cpu'), scale_factor=magnification_factor, mode='bicubic').to(self.device)

        upsampled_lr_img = self.relu(self.conv_upsampled_lr_img(upsampled_lr_img))

        x = self.relu(self.conv0(x))

        x = self.batch_norm(x + upsampled_lr_img + t)
        batch_size = x.size(0)
        x = transforms.Resize((224, 224))(x)
        x = self.vit(x)
        x = transforms.Resize((image_size, image_size))(x)
        x = self.conv(x)
        return x

if __name__ == '__main__':
    input_channels = output_channels = 3
    img_size = 256
    magnification_factor = 2
    batch_size = 8
    noise_steps = 1500
    device = 'cpu'

    model = ViTModel(image_channels=input_channels, device=device)

    input_tensor = torch.randn(batch_size, input_channels, img_size, img_size)
    t = torch.randint(low=1, high=noise_steps, size=(batch_size,))
    lr_img = torch.randn(batch_size, input_channels, img_size//magnification_factor, img_size//magnification_factor)
    # Forward pass through the model
    output_tensor = model(input_tensor,t, lr_img, magnification_factor)

    # Check output shape
    print(f"Output shape: {output_tensor.shape}")

    print("Num params: ", sum(p.numel() for p in model.parameters()))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Non-trainable parameters: {non_trainable_params}")