import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]])
dataset=torchvision.datasets.CIFAR10("./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)
output=torch.reshape(input,(-1,1,2,2))#计算batch_size=2*2/(1*2*2)
print(output.shape)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()
    def forward(self,input):
        output=self.sigmoid1(input)
        return output
model=Model()
writer=SummaryWriter("logs_sigmoid")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("sigmoid_input",imgs,step)
    output=model(imgs)
    writer.add_images("sigmoid_output",output,step)
    step=step+1
writer.close()