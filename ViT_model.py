import timm
import torch.nn as nn
from torchvision import transforms
import torch

# Define the model class
class ViTModel(nn.Module):
    def __init__(self, pretrained_model_name='vit_base_patch16_224'):
        super(ViTModel, self).__init__()
        # Load a pretrained Vision Transformer model
        self.vit = timm.create_model(pretrained_model_name, pretrained=True)
        
        # Adapt the classifier to return an image of the same shape as input (3, 256, 256)
        self.vit.head = nn.Sequential(
            nn.Linear(self.vit.head.in_features, 256 * 256 * 3),
            nn.Unflatten(1, (3, 256, 256))
        )
        
        # Add additional layers if needed for more processing
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # Reshape input to fit the Vision Transformer input requirement
        batch_size = x.size(0)
        x = transforms.Resize((224, 224))(x)
        x = self.vit(x)
        x = transforms.Resize((256, 256))(x)
        x = self.conv(x)
        return x

if __name__ == '__main__':

    # Create the model
    model = ViTModel()

    # Example input tensor
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 256, 256)

    # Forward pass through the model
    output_tensor = model(input_tensor)

    # Check output shape
    print(f"Output shape: {output_tensor.shape}")

    # %%
    print("Num params: ", sum(p.numel() for p in model.parameters()))