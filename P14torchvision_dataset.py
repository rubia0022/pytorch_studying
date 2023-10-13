import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)#./表示当前的untitle文件夹
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)

print(test_set[0])#这里数据类型是序列(ToTensor,int)
writer=SummaryWriter("p14")#将tensorboard的日志logs名称设置为p14
for i in range(10):
    img_tensor,target=test_set[i]
    print(img_tensor.shape)
    writer.add_image("test_set",img_tensor,i)
writer.close()