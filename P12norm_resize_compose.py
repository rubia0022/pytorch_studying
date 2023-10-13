from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("D:\\deep_learning_dataset\\hymenoptera_data\\train\\ants\\0013035.jpg")
#ToTensor 这是一个类，下面将PIL的Image类转换为ToTensor类并将其用tensorboard显示出来
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)
#Normalize归一化  output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])#三个均值和三个标准差，范围自设（0，1）
img_norm=trans_norm(img_tensor)#归一化output[channel] = (input[channel] - mean[channel]) / std[channel]，则img_norm=2*img_totensor-1
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

#Resize 重塑型，可输入序列（H，W）输出为H*W的图片，或者输入一位int，输出原图的等比例缩减int倍
print(img.size)
trans_resize=transforms.Resize((512,512))#将resize类进行实例化，作用是按照h=512，w=512进行输出
img_resize=trans_resize(img)
print(img_resize)#此时是PIL的image数据类型，要加入tensorboard需要将去转换为tensor类型
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)

#Compose 可以结合多个transforms，即把一个图片的多个操作结合成一次性完成
trans_resize_2=transforms.Resize(300)
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])#将Compose类进行实例化，作用是Resize和ToTensor结合起来
img_resize_2=trans_compose(img)
writer.add_image("Compose",img_resize_2,1)

#RandomCrop对图片进行裁剪的函数，如果是一位int，则裁剪为正方形，若是二位序列（H，W），则裁剪为H*W
trans_random=transforms.RandomCrop(512)#将RandomCrop的实例化，裁剪为512*512
trans_compose_2=transforms.Compose([trans_random,trans_totensor])#将Compose实例化，进行裁剪并转为totensor
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)


writer.close()
