import torch
from torch import nn, einsum
from einops import rearrange

from .rel_pos_embedding import AbsPosEmb, RelPosEmb


class MHSA(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        heads = 4,
        dim_qk = 128,
        dim_v = 128,
        rel_pos_emb = False):
        """
        dim: number of channels of feature map
        fmap_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()
        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = nn.Conv2d(dim, out_channels_qk * 2, 1, bias=False) # 1*1 conv to compute q, k
        self.to_v = nn.Conv2d(dim, out_channels_v, 1, bias=False) # 1*1 conv to compute v
        self.softmax = nn.Softmax(dim=-1)

        height, width = fmap_size
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(height, width, dim_qk)
        else:
            self.pos_emb = AbsPosEmb(height, width, dim_qk)

    def forward(self, featuremap):
        """
        featuremap: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        heads = self.heads
        B, C, H, W = featuremap.shape
        q, k = self.to_qk(featuremap).chunk(2, dim=1)
        v = self.to_v(featuremap)
        q, k, v = map(lambda x: rearrange(x, 'B (h d) H W -> B h (H W) d', h=heads), (q, k, v))

        q *= self.scale

        logits = einsum('b h x d, b h y d -> b h x y', q, k)
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = einsum('b h x y, b h y d -> b h x d', weights, v)
        attn_out = rearrange(attn_out, 'B h (H W) d -> B (h d) H W', H=H)

        return attn_out





         
        