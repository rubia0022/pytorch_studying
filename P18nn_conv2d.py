import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3)#定义一个卷积类
    def forward(self,x):
        x=self.conv1(x)#卷积类实例化，进行一次卷积操作
        return x
model=Model()
print(model)

writer=SummaryWriter("con2v_logs")
step=0
for data in dataloader:
    imgs,targets=data
    output=model(imgs)
    print(imgs.shape)
    print(output.shape)
    #torch.Size([64, 6, 30, 30])
    writer.add_images("conv2d_input",imgs,step)
    #torch.Size([64, 6, 30, 30])无法显示出6个channel，直接输出到tensorboard会报错则需要转换成[xxx, 3, 30, 30]
    output=torch.reshape(output,(-1,3,30,30))#这里(-1,3,30,30)batch_size=-1表示程序要根据后面的（,3,30,30）来自动计算，计算方式为64*6*30*30/(3*30*30)
    writer.add_images("conv2d_output", output, step)

    step=step+1
writer.close()
