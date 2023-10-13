import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,64)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1=Sequential(#Sequentia比较像compose函数，将多个操作整合到一起
            Conv2d(3, 32,5, padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,  padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,x):
        x=self.model1(x)
        return x
model=Model()
loss=nn.CrossEntropyLoss()
for data in dataloader:
    imgs,targets=data
    output=model(imgs)
    result_loss=loss(output,targets)
    result_loss.backward()