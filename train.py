import torch
from model.resnet_backbone import ResNet_with_BotStack

inp = torch.rand(1, 3, 512, 512)
botnet = ResNet_with_BotStack(fmap_size=(32, 32), botnet=True)
y = botnet(inp)
print(y.shape)