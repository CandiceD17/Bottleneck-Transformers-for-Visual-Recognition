import torch
from torch import nn

from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, fmap_size, heads = 4, dim_head = 128, rel_pos_emb = False):
        super.__init__()
        
        