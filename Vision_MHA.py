# %%
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor
from torchinfo import summary
import torch.nn.functional as F
# %%
class ImageEncoding(nn.Module):
    def __init__(self, input_channels,image_size, patch_size = (16,16), stride = None, embedding_dim:int = None):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        if stride is None:
            self.stride = patch_size
        else:
            self.stride = stride

        if embedding_dim is None:
            self.embedding_dim = input_channels * patch_size[0] * patch_size[1] # P^2*C as in the original paper
        else:
            self.embedding_dim = embedding_dim

        self.conv = nn.Conv2d(in_channels = input_channels, out_channels = self.embedding_dim, kernel_size=self.patch_size, stride=self.stride) # (batch_size, num_channel, height, widht) -> (batch_size, embedding_dim, num_patches_y, num_patches_x)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3) # (batch_size, embedding_dim, num_patches_y, num_patches_x) -> (batch_size, embedding_dim, num_patches)
    
    def forward(self, x):
        if x.shape[-1] != self.image_size:
            try:
                x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bicubic')
            except:
                x = F.interpolate(x.to('cpu'), size=(self.image_size, self.image_size), mode='bilinear').to(x.device)
        # x = (batch_size, channel, height, width)
        x = self.conv(x) # (batch_size, embedding_dim, num_patches_y, num_patches_x)
        x = self.flatten(x) # (batch_size, embedding_dim, num_patches)
        # we want (batch_size, num_patches, embedding_dim)
        x = x.transpose(1,2)
        return x
    
class PositionalEncoding_without_CLS_param(nn.Module):

    def __init__(self, image_size, embedding_dim, patch_size=(16,16)) -> None:
        super().__init__()
        # create the token embedding
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        # create the positional embedding, in the paper it is used a standard learnable 1D position embeddings
        self.num_patch = self.image_size*self.image_size // (patch_size[0]*patch_size[1])
        self.position_emb = nn.Parameter(torch.rand(1, self.num_patch, self.embedding_dim), requires_grad=True)
        
    def forward(self, x):
        x = x + self.position_emb
        return x
    
class Embedding(nn.Module):
    def __init__(self, input_channels, image_size, patch_size = (16,16), stride = None, embedding_dim=None) -> None:
        super().__init__()

        self.image_embedding = ImageEncoding(input_channels, image_size, patch_size, stride, embedding_dim)
        self.embedding_dim = self.image_embedding.embedding_dim  
        self.positional_encoding = PositionalEncoding_without_CLS_param(image_size, self.embedding_dim, patch_size)
        self.num_patches = self.positional_encoding.num_patch

    def forward(self, x):
        x = self.image_embedding(x)
        x = self.positional_encoding(x)
        return x

class MultiheadAttention(nn.Module):

    def __init__(self, head:int, embedding_dim:int, ) -> None:
        super().__init__()

        self.head = head
        self.embedding_dim = embedding_dim

        assert embedding_dim % head == 0, "embedding_dim must be divisible by head"
        self.head_dim = embedding_dim // head

        self.w_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.w_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.w_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.w_o = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def attention(self, q, k, v, mask=None):
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(attention_scores, dim=-1)

        x = torch.matmul(attention, v)
        return x

    def forward(self, q,k,v ,mask= None):
        # q,k,v = (batch_size, num_patches+1, embedding_dim)

        q1 = self.w_q(q)
        k1 = self.w_k(k)
        v1 = self.w_v(v)

        # -view->((batch_size, num_patches+1, head, head_dim)  -transpose-> (batch_size, head, num_patches+1, head_dim)
        q1 = q1.view(q1.shape[0], q1.shape[1], self.head, self.head_dim).transpose(1,2) 
        k1 = k1.view(k1.shape[0], k1.shape[1], self.head, self.head_dim).transpose(1,2)
        v1 = v1.view(v1.shape[0], v1.shape[1], self.head, self.head_dim).transpose(1,2)

        x = self.attention(q1, k1, v1, mask=mask) # (batch_size, head, num_patches+1, head_dim)
        x = x.transpose(1,2) # (batch_size, num_patches+1, head, head_dim)
        x = x.flatten(start_dim=2, end_dim=3) # (batch_size, num_patches+1, embedding_dim)

        x = self.w_o(x)
        return x

        return x
      
class Vision_MHA(nn.Module):
    def __init__(self,image_size, input_channels, patch_size, num_heads, embedding_dropout=0.1, embedding_dim=None):
        super().__init__()

        self.embedding = Embedding(input_channels, image_size, patch_size, embedding_dim = embedding_dim)
        self.embedding_dim = self.embedding.embedding_dim
        self.dropout = nn.Dropout(embedding_dropout)
        self_attention_block = MultiheadAttention(num_heads, self.embedding_dim)
        self.num_patches = self.embedding.num_patches
        self.norm = nn.LayerNorm(normalized_shape=[self.num_patches, self_attention_block.embedding_dim])
        self.self_attention_block = self_attention_block
        self.deconv = nn.ConvTranspose2d(in_channels=self.embedding_dim, out_channels=input_channels, kernel_size=patch_size, stride=patch_size) # ADJUST THE STRIDE THAT IS SPECIFIC FOR WHEN STRIDE IS NOT PROVIDED
        
    def forward(self, q,k):
        batch_size = q.shape[0]
        q = self.embedding(q)
        q = self.dropout(q)

        k = self.embedding(k)
        k = self.dropout(k)
        
        q = self.norm(q)
        k = self.norm(k)

        x = self.self_attention_block(q,k,k)
        x = x.transpose(1,2)

        reverse_x = x.view(batch_size, self.embedding_dim, int(self.num_patches**0.5), int(self.num_patches**0.5))
    
        deconv_reverse_x = self.deconv(reverse_x)
        return deconv_reverse_x
    

if __name__ == '__main__':
    import torch
    import torch.nn as nn


    image_size = (224,224)
    input_channels = 3
    patch_size = (16,16)
    batch_size = 64
    num_heads = 4
    device = 'mps'

    model = Vision_MHA(image_size, input_channels, patch_size, batch_size, num_heads, embedding_dropout=0.1, embedding_dim=None).to(device)
    x = torch.randn((batch_size, input_channels, image_size[0], image_size[1])).to(device)

    VMHA_output = model(x,x)

    print(x.shape)
    print(VMHA_output.shape)

    # reverse_VMHA = VMHA_output.view(batch_size, 768, 14, 14)  
    # deconv = nn.ConvTranspose2d(in_channels=768, out_channels=3, kernel_size=patch_size, stride=patch_size[0]).to(device)
    # deconv_VMHA = deconv(reverse_VMHA)
    # print(deconv_VMHA.shape)

    # summary(model, input_size=(64,1, 28, 28))
# %%
