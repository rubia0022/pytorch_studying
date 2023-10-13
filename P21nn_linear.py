import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64,drop_last=True)#当batch_size为64时，最后一项只有16个图像，不足64，要舍去，否则self.linear1=Linear(196608,10)会报错

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1=Linear(196608,10)
    def forward(self,input):
        output=self.linear1(input)
        return output
model=Model()
for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    output=torch.reshape(imgs,(1,1,1,-1))
    print(output.shape)
    output=model(output)
    print(output.shape)