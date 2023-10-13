import torchvision
from torch.utils.data import DataLoader

#准备测试集
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)


print(len(test_data))
print(len(test_loader))#test_loader将test_data每4个进行打包
img,target=test_data[0]
print(img.shape)
print(target)
#imgs,targets=test_loader[0]   报错，由于test_loader类型不可根据索引指定
for data in test_loader:
     imgs,targets=data
     print(imgs.shape)#由于输出均为torch.Size([4, 3, 32, 32])表示4个set打包，3通道，图片大小均为32*32，则只有一个打包
     print(targets)#输出假设为tensor([1, 5, 6, 6])表示这几个dataset的target分别是1，5，6，6
     print("\n")