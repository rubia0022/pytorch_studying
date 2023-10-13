import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)
    def forward(self,input):
        output=self.maxpool1(input)
        return output
model=Model()
writer=SummaryWriter("logs_maxpool")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("maxpool_input",imgs,step)
    output=model(imgs)#区别于卷积，池化不改变通道数out_channel，所以这里不需要像卷积那样对output进行reshape
    writer.add_images("output_maxpool",output,step)
    step=step+1
writer.close()