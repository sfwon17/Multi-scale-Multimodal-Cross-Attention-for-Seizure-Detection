import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from vit_pytorch import ViT, SimpleViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class EGG_Multiscale_CA(nn.Module):
    def __init__(self, *, image_size, large_patch_size, small_patch_size, num_classes, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        small_patch_height, small_patch_width = pair(small_patch_size)
        large_patch_height, large_patch_width = pair(large_patch_size)
        
        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // small_patch_height) * (image_width // small_patch_width)
        #print(num_patches)
        small_patch_dim =  9 * small_patch_height * small_patch_width
        large_patch_dim =  9 * large_patch_height * large_patch_width
        #print(patch_dim)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # alpha for cross attention
        self.alpha_short = nn.Parameter(torch.full((640, 1), 0.5))
        self.alpha_long = nn.Parameter(torch.full((32, 1), 0.5))  # Assuming 32 patches for long-time
        
        # alpha for class token
        self.cls_alpha_tokens = nn.Parameter(torch.tensor(0.5))
        self.cls_alpha_tokens2 = nn.Parameter(torch.tensor(0.5))
        # make sure embedding match for short to long
        self.short_to_long = nn.Sequential(
            nn.Linear(640, 256),  # First layer (you can adjust this intermediate size)
            nn.ReLU(),            # Non-linear activation
            nn.Linear(256, 32)    # Second layer
        )
        
        # make sure embedding match for long to short
        self.long_to_short = nn.Sequential(
            nn.Linear(32, 256),   # First layer
            nn.ReLU(),            # Non-linear activation
            nn.Linear(256, 640)   # Second layer
        )

        
        # short-time
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(1, 3), stride=(1, 1),padding="same")
        self.layernorm1 = nn.BatchNorm2d(3)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(3, 6, kernel_size=(1, 5), stride=(1, 1),padding="same")
        self.layernorm2 = nn.BatchNorm2d(6)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(6, 9, kernel_size=(1, 10), stride=(1, 1),padding="same")
        self.layernorm3 = nn.BatchNorm2d(9)
        self.relu3 = nn.ReLU()
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 =small_patch_height, p2 = small_patch_width),
            nn.LayerNorm(small_patch_dim),
            nn.Linear(small_patch_dim, 128),
            nn.LayerNorm(128),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, 128))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(128, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_mid = nn.Linear(128, 16)
        
        # long-time
        self.conv4 = nn.Conv2d(1, 3, kernel_size=(1, 10), stride=(1, 1),padding="same")
        self.layernorm4 = nn.BatchNorm2d(3)
        self.relu4 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(3, 6, kernel_size=(1, 15), stride=(1, 1),padding="same")
        self.layernorm5 = nn.BatchNorm2d(6)
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(6, 9, kernel_size=(1, 20), stride=(1, 1),padding="same")
        self.layernorm6 = nn.BatchNorm2d(9)
        self.relu6 = nn.ReLU()

        self.to_patch_embedding2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = large_patch_height, p2 = large_patch_width),
            nn.LayerNorm(large_patch_dim),
            nn.Linear(large_patch_dim, 128),
            nn.LayerNorm(128),
        )

        self.pos_embedding2 = nn.Parameter(torch.randn(1, num_patches + 1, 128))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, 128))
        self.dropout2 = nn.Dropout(emb_dropout)
        self.transformer2 = Transformer(128, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool2 = pool
        self.to_latent2 = nn.Identity()

        self.mlp_mid2 = nn.Linear(128, 16)

        #fusion 
        self.transformer3 = Transformer(128, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool3 = pool
        self.to_latent3 = nn.Identity()

        self.mlp_mid3 = nn.Linear(128, 16)
        
        # final
        self.mlp_mid5 = nn.Linear(48,8)
        self.mlp_head = nn.Linear(8,2)
        
    def forward(self, img):

        ### short time ### 
        x = self.conv1(img)
        x = self.layernorm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.layernorm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.layernorm3(x)
        x = self.relu3(x)
        
        x = self.to_patch_embedding(x)
        original_x_embedding = x.clone () # this is for cross attention later on
        
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        embeded_x = x.clone()
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_mid(x)
        
        # long time
        y = self.conv4(img)
        y = self.layernorm4(y)
        y = self.relu4(y)
        
        y = self.conv5(y)
        y = self.layernorm5(y)
        y = self.relu5(y)
        
        y = self.conv6(y)
        y = self.layernorm6(y)
        y = self.relu6(y)

        # process through transformer #
        y = self.to_patch_embedding2(y)
        original_y_embedding = y.clone () # this is for cross attention later on
        b, n, _ = y.shape
        cls_tokens2 = repeat(self.cls_token2, '1 1 d -> b 1 d', b = b)
        y = torch.cat((cls_tokens2, y), dim=1)
        y += self.pos_embedding2[:, :(n + 1)]
        y = self.dropout2(y)
        y = self.transformer2(y)
        embeded_y = y.clone()
        y = y.mean(dim = 1) if self.pool2 == 'mean' else y[:, 0]
        y = self.to_latent2(y)
        y = self.mlp_mid2(y)

        # get process token
        cls_tokens = embeded_x[:, :1, :]
        embeded_x = embeded_x[:, 1:, :]
        
        cls_tokens2 = embeded_y[:, :1, :]
        embeded_y = embeded_y[:, 1:, :]

        ## cross attention with fusion for cls with weights
        cls_tokens_sum1 = self.cls_alpha_tokens * cls_tokens + self.cls_alpha_tokens2 * cls_tokens2
        
        # cross attention * original patch with weighted sum 
        x_sum = self.alpha_short * embeded_x + (1 - self.alpha_short) * original_x_embedding 
        y_sum = self.alpha_long * embeded_y + (1 - self.alpha_long) * original_y_embedding
        
        xy_sum = torch.cat((cls_tokens_sum1, x_sum, y_sum), dim=1)    
        xy_sum = self.transformer3(xy_sum)     
        xy_sum = xy_sum.mean(dim = 1) if self.pool3 == 'mean' else x_sum[:, 0]

        xy_sum = self.to_latent3(xy_sum) 
        xy_sum = self.mlp_mid3(xy_sum)
        
        # concate results
        xyz = torch.cat((x, y, x_sum), dim=1)
        xyz = self.mlp_mid5(xyz)
        xyz = self.mlp_head(xyz)
        return xyz
