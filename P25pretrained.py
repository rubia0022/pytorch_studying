import torchvision

#train_data=torchvision.datasets.ImageNet("./data_imageNet",split='train',download=True,transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false=torchvision.models.vgg16(weights=None)#默认没有预训练，如果需要进行预训练则把参数weights=‘default’
vgg16_true=torchvision.models.vgg16(weights='DEFAULT')
print(vgg16_true)

vgg16_true.add_module('add_linear',nn.Linear(1000,10))#给原vgg16模型添加一层线性变换
print(vgg16_true)

vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))#在vgg16模型的classfier里添加线性变换
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6]=nn.Linear(4096,10)#修改vgg16false的模型6号层
print(vgg16_false)