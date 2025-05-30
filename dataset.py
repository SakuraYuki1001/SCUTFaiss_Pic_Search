'''
这是一个用于处理数据集的模块，主要功能包括：
1. 获取数据集的路径。
2. 定义一个自定义数据集类MyDataset，继承自torch.utils.data.Dataset。
3. 在MyDataset类中，读取数据集中的图像文件，并进行预处理。
'''

# 导入必要的库
import os
from torch.utils.data import Dataset  # 用于数据集处理
from PIL import Image  # 用于图像处理

'''
自定义 MyDataset 类，用于加载和处理图像数据集。

Arguments(参数说明):
    data_path (str): 数据集路径。
    transform (callable, optional): 图像预处理函数。
'''
class MyDataset(Dataset):  # 自定义数据集类，继承自 PyTorch 的 Dataset 类
    '''1.构造函数'''
    def __init__(self, data_path, transform=None):
        super().__init__() # 调用父类的构造函数
        self.transform = transform  # 图像预处理函数
        self.data_path = data_path  # 数据集路径
        self.data = []  # 初始化一个空列表来保存图像文件名

        img_path = os.path.join(data_path, 'aimg.txt')  # 组合图像文件列表路径
        with open(img_path, 'r', encoding='utf-8') as f:  # 打开图像文件列表
            '''
            注意：aimg中应该有我们自己填写图片文件名
            '''
            for line in f.readlines():  # 读取每一行图像文件名
                line = line.strip()  # 去掉行末尾的空格和换行符
                img_name = os.path.join(data_path, line)  # 完整图像路径

                img = Image.open(img_name)  # 打开图像文件
                if img.mode == 'RGB':  # 如果图像是 RGB 模式
                    self.data.append(line)  # 将图像文件名加入到数据集中

    '''2.获取数据集中的一个样本'''
    def __getitem__(self, idx):  # 获取数据集中的一个样本，该样本是通过索引 idx 获取的
        # 获取当前索引对应的图像路径
        img_path = os.path.join(self.data_path, self.data[idx])  # 当前索引对应的图像路径
        # 读取图像
        img = Image.open(img_path)  # 打开图像文件
        # 应用预处理函数
        if self.transform:  # 如果定义了预处理函数
            img = self.transform(img)  # 应用预处理函数
        # 返回图像和索引
        dict_data = {  # 将数据转换成一个索引和数据对应的字典
            'index': idx,
            'img': img
        }

        return dict_data  # 返回字典数据

    '''3.获取数据集的长度'''
    def __len__(self):  # 返回数据集的长度
        return len(self.data)