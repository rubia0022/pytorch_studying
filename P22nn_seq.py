import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.maxpool1=MaxPool2d(2)
        self.conv2=Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.maxpool2=MaxPool2d(2)
        self.conv3=Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten=Flatten()#将多维向量展开为1维，等同于reshape(1,-1,)
        self.linear1=Linear(1024,64)
        self.linear2=Linear(64,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        return x
model=Model()
print(model)
input=torch.ones((64,3,32,32))#生成batch_size=64的3*32*32的数值均为1的dataloader数据
output=model(input)
print(output.shape)