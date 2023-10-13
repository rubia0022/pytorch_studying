import torch
from torch import nn

#加载自己的模型
class Model(nn.Module):#定义模型
    def __init__(self):
        super(Model, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3)
    def forward(self,x):
        x=self.conv1(x)
        return x
model=Model()
torch.save(model,"model_method.pth")#保存模型

model=torch.load("model_method.pth")#加载模型
print(model)