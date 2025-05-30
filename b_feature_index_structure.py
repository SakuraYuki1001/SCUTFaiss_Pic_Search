'''
构建图片特征索引——设置数据格式
1.基于图片特征建立索引，便于快速检索
2.这里我们使用Faiss开源向量的索引库构建图片特征索引(检索速度很快)
3.索引格式<特征,索引Id>
'''
from SCUTFaiss_Pic_Search.a_test import img2feat
from theory import *  #从theory模块导入img2feat函数，用于提取图像特征
import numpy as np #使用numpy库来处理数组和矩阵运算
ids = []  #初始化一个空列表ids，用于存储图像索引
data = []  #初始化一个空列表data，用于存储图像特征

img_path = os.path.join(img_folder, 'aimg.txt')  # 图像文件列表路径，其中都是图像的文件名
with open(img_path, 'r', encoding='utf-8') as f:  # 打开图像文件列表
    for line in f.readlines():  # 遍历每一行图像文件名
        
        '''这里的line是读取文件f中的一行文本，表示图像的文件名。上下两种方法都行
        line = line.strip()  # 去掉行末尾的空格和换行符
        img_name = os.path.join(img_folder, line + '.txt')  # 完整图像路径
        '''
        img_name = line.strip()   #这一行代码的作用是读取文件 f 中的一行文本，然后去除字符串两端的空格，赋值给变量 img_name。该代码是读取图片文件名的操作。
        img_id = img_name.split('.')[0]   #将图片名称（例如'cat.jpg'）按照"."进行分割，获取列表['cat', 'jpg']，然后取第一个元素'cat'作为图片的id。
        pic_txt_file = os.path.join( img_folder,"{}.txt".format(img_name) )   #这行代码是将当前图片的文件名和文件夹路径连接起来，构造出该图片所对应的标注文本文件的路径。

        if not os.path.exists(pic_txt_file):
            print(f"Warning: {pic_txt_file} does not exist.")
            continue

        feat = img2feat(pic_txt_file)  #得到特征
        ids.append(int(img_id))  #得到图片id
        data.append(np.array(feat))  #得到图片特征

# 构建数据集<id,data>
ids - np.array(ids)  #将ids列表转换为NumPy数组
data = np.array(data).astype('float32')  #将data列表转换为NumPy数组，并将数据类型转换为float32
d = 2048 #feature 特征长度(模型结果)，与实际特征维度一致
print("特征向量记录数据:", data.shape)  # 是含有图片数据1000，其中每张图片记录了2048个特征
# print("特征向量记录id:", ids.shape)  # 是含有图片数据1000，其中每张图片记录了512个特征