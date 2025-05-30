'''
Pytorch：数据预处理，通过预训练网络提取特征，修改网络模型
自定义Dataset：PictureVOC
通过DataLoader，批量加载我们的Dataset
加载我们的图片，将项链话数据加载并保存为文本文件
'''

#导入库
import os #需要使用系统信息
from PIL import Image #PIL库用于处理图像
import torch #torch库用于深度学习
from torch.utils.data import DataLoader #导入torch.utils.data，只需要DataLoader类，Dataset我们是自定义的
from torchvision import models #从torchvision库导入已经训练好的模型，用于图像处理

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #检查是否有GPU可用，如果有则使用GPU，否则使用CPU
# print("【theory】正在使用的设备是:", device) #打印当前使用的设备



'''
初始化模型和定义并编写特征提取的方法函数
'''
#使用外部网路模型,使用预训练的ResNet50模型
from tools import load_model,feature_extract,transform
model = load_model() #载入模型



'''
加载数据
1.自定义数据库，继承一个Dataset类
2.验证自定义的数据集是否可以正常加载
3.使用DataLoader验证
'''
from dataset import MyDataset #从dataset模块导入自定义的数据集类MyDataset

#实现我们的DataLoader，加载数据
img_folder = 'SCUTFaiss_Pic_Search\PictureXyy'  # 数据集路径（一定要相对路径！）

# 创建自定义的数据集PictureVOC实例：val_dataset，将数据集文件夹路径和自定义transform预处理函数传递给MyDataset类的构造函数。
# 即可得到一个包含处理后的含有图像和相对应索引值的数据集对象。
val_dataset = MyDataset( img_folder, transform=transform)

# 创建DataLoader实例：val_dataloader，用于批量加载数据集。
batch_size = 64  # 设置批处理的图像数量
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # 创建DataLoader实例，用于批量加载数据集

'''
print("【theory】dataset中的图片数量:", val_dataset.__len__())  # 打印数据集中的图像数量
print("【theory】迭代器:", int(val_dataset.__len__()/batch_size)+1)  
以上是计算并输出迭代次数，即在每个epoch中需要迭代的次数
其中，val_dataset.__len__()返回数据集的长度，batch_size是每个批次的图像数量，除以batch_size后加1是为了确保即使最后一个批次的图像数量不足batch_size也能被处理。
'''

'''
# 查看每个批batch的数据情况，包括索引和图像
if __name__ == '__main__':
    # 查看每个批batch的数据情况，包括索引和图像
    batch = next(iter(val_dataloader))  
    print("【theory】Batch keyspi批处理键值对:", batch.keys())  # 打印批次数据的键值对
    # 获取了val_dataloader中的第一个批次数据
    # 其中inter()函数是DataLoader返回的迭代器，netx()函数用于获取下一个batch数据，然后用keys()函数查看batch中的键值对。

    img = batch['img']  # 获取img对应的数据
    index = batch['index']  # 获取batch这个key对应的数据

    print("【theory】",img.shape)  # 打印出来的依次是batch的大小、通道数、图像高度和宽度
    print("【theory】",index)  # 打印索引值
'''