import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#准备测试集


test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)#这里改为64


img,target=test_data[0]
print(img.shape)
print(target)
writer=SummaryWriter("dataloader")
step=0
#imgs,targets=test_loader[0]   报错，由于test_loader类型不可根据索引指定
for data in test_loader:
     imgs,targets=data
     writer.add_images("test_load",imgs,step)#注意这里使用的add_images而不是add_image
     step=step+1
writer.close()