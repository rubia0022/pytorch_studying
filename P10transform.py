from PIL import Image
from torchvision import transforms
#python的用法->tensor数据类型
#可以将数据类型PIL Image或numpy.ndarray转换为tensor数据类型
#通过transform.ToTensor去看两个问题：
# 1.transform该如何使用（python）
# 2.为什么我们需要Tensor数据类型
img_path="D:\\deep_learning_dataset\\hymenoptera_data\\train\\ants\\0013035.jpg"
img=Image.open(img_path)
# 1.transform该如何使用（python）
tensor_trans=transforms.ToTensor()#将transforms中的ToTensor类实例化
tensor_img=tensor_trans(img)
print(tensor_img)