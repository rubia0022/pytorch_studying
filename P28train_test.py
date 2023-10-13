import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备数据集
train_data=torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
#求数据集长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#加载为dataloader数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#搭建神经网络
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

#创建网络模型
model=Model()
#损失函数
loss_fn=nn.CrossEntropyLoss()

#优化器
learning_rate=1e-2 #1*10^(-2)=0.01
optimizier=torch.optim.SGD(model.parameters(),lr=learning_rate)
#设置训练网络的一些参数
total_train_step=0#记录训练的次数
total_test_step=0#记录测试的次数
epoch=10#训练的轮次

#添加tensorboard
writer=SummaryWriter("logs_train_test")
#开始训练
for i in range(epoch):
    print("--------第{}轮训练开始------".format(i+1))
    #训练步骤开始
    for data in train_dataloader:
        imgs,targets=data
        outputs=model(imgs)
        loss=loss_fn(outputs,targets)
        #优化器优化模型
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()

        total_train_step+=1
        if total_train_step%100==0:#每训练100步才打印一次loss，防止输出太多太乱
            print("训练次数：{}，loss：{}".format(total_train_step,loss.item()))#item将loss的数据类型转换为只有loss的值的数值型
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试步骤
    total_test_loss=0#往往训练需要求本次epoch训练的模型在所有测试集上的loss
    with torch.no_grad():#在已有的模型基础上设置梯度为0，即不再调优
        for data in test_dataloader:
            imgs,targets=data
            outputs=model(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss+=loss.item()
    print("整体测试集上的loss：{}".format(total_test_loss))
    total_test_step += 1
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    #保存每epoch轮的模型
    torch.save(model,"model_{}.pth".format(i))
    print("模型已保存")
writer.close()

