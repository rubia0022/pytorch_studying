import torch
from torch import nn
from torch.nn import ReLU

input=torch.tensor([[1,-0.5],
                    [-1,3]])

output=torch.reshape(input,(-1,1,2,2))#计算batch_size=2*2/(1*2*2)
print(output.shape)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.relu1=ReLU()
    def forward(self,input):
        output=self.relu1(input)
        return output
model=Model()
output=model(input)
print(output)