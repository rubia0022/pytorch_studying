import torch
from torch import nn
from torch.nn import L1Loss

inputs=torch.tensor([1,2,3],dtype=torch.float32)#L1Loss函数的输入变量应该是浮点型，否则报错
targets=torch.tensor([1,2,6],dtype=torch.float32)

inputs=torch.reshape(inputs,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))
#使用L1Loss  绝对值损失
loss=L1Loss(reduction="sum")
result=loss(inputs,targets)
print(result)
#使用MSELoss  方差损失
loss_mse=nn.MSELoss()
result_mse=loss_mse(inputs,targets)
print(result_mse)
#交叉熵，常用于分类的函数，优化器有一个优化函数，最小二乘法，不同的优化函数对应不同的损失函数，
# mse是经典用于回归的损失函数
x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))#CrossEntropyLoss的输入是（N，C）格式，N是batch_size，C区别之前的卷积函数（通道），这里指类别数
loss_cross=nn.CrossEntropyLoss()
result_cross=loss_cross(x,y)
print(result_cross)
