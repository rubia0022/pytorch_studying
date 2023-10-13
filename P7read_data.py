from torch.utils.data import Dataset
from PIL import Image
import os
class MyData(Dataset):#__xx__类型的函数是python的类中必有的函数，相当于java中的构造和析构函数
    def __init__(self,root_dir,label_dir):#java里的构造函数
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)#join的功能是将两个参数的str拼接起来
        self.img_path=os.listdir(self.path)#将self.path中的文件名称整合成一个列表list
    def __getitem__(self, idx):#类数据中单个变量的返回类型，例如实例化mydata=MyData(Dataset),使用mydata[idx]则调用此函数
        img_name=self.img_path[idx]#获取list中的某一个图片的文件名
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)

root_dir="D:\\deep_learning_dataset\\hymenoptera_data\\train"
ants_label_dir="ants"
bees_label_dir="bees"
ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)
train_dataset=ants_dataset+bees_dataset#训练集

