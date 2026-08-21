import torch
import torch.nn as nn
from models.layers import StochasticDepth

class PatchEmbedding(nn.Module):
    def __init__(self, inchannels=3, patchdim=16, embeddim=768, imagedim=224):
        super().__init__()
        self.patchdim = patchdim
        self.patchcount = (imagedim // patchdim) **2
        # projection to embedding dimensions
        self.projection = nn.Conv2d(in_channels=inchannels, out_channels=embeddim, kernel_size=patchdim, stride=patchdim)
        # initialise CLS token as learnable param
        self.CLS = nn.Parameter(torch.randn(1, 1, embeddim))
        # initialise positional embeddings for all tokens including CLS token
        self.posembed = nn.Parameter(torch.randn(1, self.patchcount+1, embeddim))

    def forward(self, x):
        # initial shape: [batchsize, channels, height, width]
        B = x.shape[0]
        # after projecting to 768 dimensions, flatten from index 2 onwards, and swap positions to get [batchsize, patchcount, embeddim]
        x = self.projection(x).flatten(2).transpose(1,2)
        # create CLS token for each image in batch
        cls_tokens = self.CLS.expand(B, -1, -1)
        # prepend CLS token to patch embedding (concat along dim 1 which is sequence length)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embeddings
        x += self.posembed
        return x

class ViTBlock(nn.Module):
    def __init__(self, embeddim=768, heads=12, drop_path_prob=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embeddim)
        self.ln2 = nn.LayerNorm(embeddim)
        self.attn = nn.MultiheadAttention(embed_dim=embeddim, num_heads=heads, batch_first=True)
        self.mlp = nn.Sequential(
            # MLP size as described in the paper
            nn.Linear(embeddim, 4*embeddim),
            nn.GELU(),
            nn.Linear(4*embeddim, embeddim),
        )
        self.stochastic_depth = StochasticDepth(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x += self.stochastic_depth(attn_out)
        mlp_out = self.mlp(self.ln2(x))
        x += self.stochastic_depth(mlp_out)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, classes, drop_path_prob, inchannels=3, patchdim=16, embeddim=768, imagedim=224, heads=12, transformer_depth=12):
        super().__init__()
        self.patchembed = PatchEmbedding(inchannels, patchdim, embeddim, imagedim)
        dp_probs = [y.item() for y in torch.linspace(0, drop_path_prob, transformer_depth)]
        self.blocks = nn.Sequential(*[ViTBlock(embeddim, heads, dp_probs[i]) for i in range(transformer_depth)])
        self.ln = nn.LayerNorm(embeddim)
        self.CLS_head = nn.Linear(embeddim, classes)

    def forward(self, x):
        x = self.patchembed(x)
        x = self.blocks(x)
        CLS = x[:, 0]
        return self.CLS_head(self.ln(CLS))
