import torch
import torchvision
from PIL import Image
from torch import nn

image_path="./img/55.png"
image=Image.open(image_path)
image=image.convert('RGB')#png图像是4通道，多了一个透明通道，需要转换为RGB的三通道
print(image)

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),#resize图像大小
                                           torchvision.transforms.ToTensor()])#将PIL的image格式转换为tensor格式
image=transform(image)
print(image.shape)

#拷贝网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.model(x)
        return x
model=torch.load("model_0.pth")#加载训练得到的网络模型
print(model)
image=torch.reshape(image,(1,3,32,32))#模型输入需要是4维，加入一个batch_size
model.eval()
with torch.no_grad():
    output=model(image)
print(output)#输出结果是0.7810比较大
print(output.argmax(1))
