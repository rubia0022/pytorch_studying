import torch
from torch import nn


class Model(nn.Module):
    def __init(self):
        super(Model, self).__init__()
    def forward(self,input):
        output=input+1
        return output

model=Model()
x=torch.tensor(1.0)
output=model(x)#这里Model父类中的nn.Module中的魔术方法call函数实现_call_impl函数会自动调用forward函数
print(output)