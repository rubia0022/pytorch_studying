import torch
import torchvision
from torch import nn

vgg16=torchvision.models.vgg16()

#保存方式1
torch.save(vgg16,"vgg16_methond1.pth")#模型保存
model=torch.load("vgg16_methond1.pth")#模型加载
print(model)

#保存方式2保存为字典格式
torch.save(vgg16.state_dict(),"vgg16_methond2.pth")#模型保存
model=torch.load("vgg16_methond2.pth")#模型加载
print(model)#输出为字典类型
#将字典类型改为网络模型
vgg16.load_state_dict(model)#通过python中的字典形式加载数据
print(vgg16)
