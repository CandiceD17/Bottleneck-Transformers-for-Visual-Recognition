import torch
from torch import nn, einsum

from einops import rearrange


def expand_dim(t, dim, k):
    """
    Expand dims for t at dim to k
    """
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    """
    x: [B, Nh * H, L, 2L - 1]
    Convert relative position between the key and query to their absolute position respectively.
    Tensowflow source code in the appendix of: https://arxiv.org/pdf/1904.09925.pdf
    """
    B, Nh, L, _ = x.shape
    # pad to shift from relative to absolute indexing
    col_pad = torch.zeros((B, Nh, L, 1)).cuda()
    x = torch.cat((x, col_pad), dim=3)
    flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
    flat_pad = torch.zeros((B, Nh, L - 1)).cuda()
    flat_x = torch.cat((flat_x, flat_pad), dim=2)
    # Reshape and slice out the padded elements
    final_x = torch.reshape(flat_x, (B, Nh, L + 1, 2 * L - 1))
    return final_x[:, :, :L, L - 1:]


def relative_logits_1d(q, rel_k):
    """
    q: [B, Nh, H, W, d]
    rel_k: [2W - 1, d]
    Computes relative logits along one dimension.
    The details of relative position is explained in: https://arxiv.org/pdf/1803.02155.pdf
    """
    B, Nh, H, W, _ = q.shape
    rel_logits = torch.einsum('b n h w d, m d -> b n h w m', q, rel_k)
    # Collapse height and heads
    rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
    rel_logits = expand_dim(rel_logits, dim=3, k=H)
    return rel_logits


class AbsPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        assert height == width
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits


class RelPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        assert height == width
        scale = dim_head ** -0.5
        self.fmap_size = height
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h = w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h