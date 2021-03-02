import torch
from model.botnet import BoTStack

inp = torch.rand(1, 2048, 32, 32)
bottleneck= BoTStack(dim=2048, fmap_size=(32, 32), stride=1, rel_pos_emb=True)
y = bottleneck(inp)
print(y)