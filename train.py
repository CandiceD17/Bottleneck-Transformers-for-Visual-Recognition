import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 256
EPOCHS = 5
IMAGE_DIR = "./ImageNet/"

def train(net, dataloader, optimizer, epoch, device):
   print("            =======  Training  ======= \n")
   net.train()
   criterion = nn.CrossEntropyLoss()
   
   for ep in range(0, epoch):
       train_loss = correct = total = 0
       
       for idx, (inputs, targets) in enumerate(train_loader):
           inputs, targets = inputs.to(device), targets.to(device)
           outputs = net(inputs)
           
           loss = criterion(outputs, targets)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
           total += targets.size(0)
           correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()
           
           if ((idx + 1) % 20 == 0 or (idx + 1) == len(train_loader)):
                print(
                    "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        ep,
                        epoch,
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )
                
     print("\n            =======  Training Finished  ======= \n")

   
if __name__ == "__main__":
    device = 'cuda'
    epoch = EPOCHS
    
    """
    Load Dataset
    """
    traindir = os.path.join(IMAGE_DIR, "train")
    trainset = torchvision.datasets.ImageFolder(
        root=traindir,
        transform=trasforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    
    """
    define network
    """
    net = ResNet_with_BotStack(fmap_size=(224, 224), botnet=True)
    net.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=1e-4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    
    train(net, train_loader, optimizer, epoch, device)
