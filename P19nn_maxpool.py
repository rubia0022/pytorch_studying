import torch
from torch import nn
from torch.nn import MaxPool2d

input =torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]],dtype=torch.float32)#最大池化函数max_pool2d要求input输入的函数值为浮点数，否则会默认为整数long类型而报错
input=torch.reshape(input,(-1,1,5,5))#将二维的input调整为4维的input以满足maxpool函数的输入shape，
# 序列的batch_size=-1表示reshape函数可以根据后面的channel=1，H=5，W=5来计算batch_size的值
print(input)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)
    def forward(self,input):
        output=self.maxpool1(input)
        return output
model=Model()
output=model(input)
print(output)

