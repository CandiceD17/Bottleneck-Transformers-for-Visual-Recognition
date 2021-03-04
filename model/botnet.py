import torch
from torch import nn, einsum
from einops import rearrange

from .self_attention_block import MHSA


"""Bottleneck Transformer (BoT) Block."""
class BoTBlock(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out,
        stride = 1,
        heads = 4,
        proj_factor = 4,
        dim_qk = 128,
        dim_v = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()):
        """
        dim: channels in feature map
        dim_out: output channels for feature map
        """
        super().__init__()
        if dim != dim_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()
        
        bottleneck_dimension = dim_out // proj_factor # from 2048 to 512
        attn_dim_out = heads * dim_v

        self.net = nn.Sequential(
            nn.Conv2d(dim, bottleneck_dimension, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_dimension),
            activation,
            MHSA(
                dim = bottleneck_dimension,
                fmap_size = fmap_size,
                heads = heads,
                dim_qk = dim_qk,
                dim_v = dim_v,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if stride == 2 else nn.Identity(), # same padding
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        nn.init.zeros_(self.net[-1].weight) # last batch norm uses zero gamma initializer
        self.activation = activation
    
    def forward(self, featuremap):
        shortcut = self.shortcut(featuremap)
        featuremap = self.net(featuremap)
        featuremap += shortcut
        return self.activation(featuremap)


"""c5 Blockgroup of BoT Blocks."""
class BoTStack(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out = 2048,
        heads = 4,
        proj_factor = 4,
        num_layers = 3,
        stride = 2, 
        dim_qk = 128,
        dim_v = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        """
        dim: channels in feature map
        fmap_size: [H, W]
        """
        super().__init__()

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)

            fmap_divisor = (2 if stride == 2 and not is_first else 1)
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(BoTBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                stride = stride if is_first else 1,
                heads = heads,
                proj_factor = proj_factor,
                dim_qk = dim_qk,
                dim_v = dim_v,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim
        assert h == self.fmap_size[0] and w == self.fmap_size[1]
        return self.net(x)

