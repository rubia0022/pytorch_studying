import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs")
image_path="D:\\deep_learning_dataset\\hymenoptera_data\\train\\ants\\5650366_e22b7e1065.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)#将PIL图片类型转换为numpy数据类型，这样就可以在add_image中使用了
writer.add_image("test",img_array,1,dataformats='HWC')
writer.add_image("test",img_array,2,dataformats='HWC')
writer.add_image("test",img_array,6,dataformats='HWC')
#y=x
for i in range(100):
    writer.add_scalar("y=x",i,i)
writer.close()