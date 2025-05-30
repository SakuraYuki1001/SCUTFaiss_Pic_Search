from theory import *
from c_picture_search import *

from skimage.util import img_as_ubyte
from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt
import os


'''
互联网图片检索
'''
#!pip install scikit-image==0.16.2-------该库图像读取、变换、过滤、分割、特征提取
topK = 10  #需要返回20张图片
# 狗
img_src = '16.jpg'#————————————————————————————————————————————————————
from skimage import io   
img = io.imread('SCUTFaiss_Pic_Search/PictureXyy/'+img_src)   #imread函数读取了一个图像文件，该函数返回一个numpy数组类型的图像
img = Image.fromarray(img)  #ndarray_image为原来的numpy数组类型的输入，将numpy数组类型的图像转换为Image对象类型，使用了Pillow库中的Image模块
query_img = img  #Image对象赋值给query_img变量

img = transform(img) # torch.Size([3, 224, 224])  #将图像转换为大小为3x224x224的张量
img = img.unsqueeze(0) #使用了PyTorch的unsqueeze函数，将维度为1的新维度添加到张量的第一维，即将大小为[3, 224, 224]的张量变为大小为[1, 3, 224, 224]的张量
img = img.to(device)  #将张量移动到指定的设备（如CPU或GPU）上进行处理
with torch.no_grad():   #(简单说就是不计算模型的梯度，因为梯度是为了求权重的)使用no_grad函数可以临时禁用梯度计算，以加快模型的推理速度。同时，这也可以避免在推理时意外修改模型的权重。使用no_grad函数可以临时禁用梯度计算，以加快模型的推理速度。同时，这也可以避免在推理时意外修改模型的权重。
    print('1.图片特征提取')
    # 图片->提取特征
    feature = feature_extract(model, img)
    feature_list = feature.data.cpu().tolist()[0]  #使用了cpu函数将其移动到CPU设备上。然后，使用了tolist函数将特征转换为Python列表格式，并取出其中的第一个元素。由于特征通常是一个大小为[1, n]的张量，因此使用[0]索引可以获取特征列表中的唯一元素，即大小为[n]的一维列表。
    # 特征-> 检索
    print('2.基于特征的检索，从faiss获取相似度图片')
    #print(feature_list)
    dis,ind = index_search(feature_list,topK=topK)
    #print(dis)
    print(ind)#打印最相似的索引

    print('3.图片可视化展示')
    # 拼接原图和检索结果为一张大图，带坐标轴并保存
    cols = 4
    rows = int(topK / cols)
    fig, axes = plt.subplots(rows, cols+1, figsize=(5*(cols+1), 5*rows))

    # 第一列放原图
    for row in range(rows):
        if row == 0:
            axes[row, 0].imshow(query_img)
            axes[row, 0].set_title('Query', fontsize=16)
        else:
            axes[row, 0].axis('off')
        axes[row, 0].set_xlabel('X')
        axes[row, 0].set_ylabel('Y')

    # 后面放检索结果
    idx = 0
    for row in range(rows):
        for col in range(1, cols+1):
            if idx < topK:
                _id = ind[0][idx]
                img_path = f'SCUTFaiss_Pic_Search/PictureXyy/{_id}.jpg'
                if os.path.exists(img_path):
                    img = io.imread(img_path)
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(f'Match {_id}', fontsize=12)
                    axes[row, col].set_xlabel('X')
                    axes[row, col].set_ylabel('Y')
                else:
                    axes[row, col].axis('off')
                idx += 1
            else:
                axes[row, col].axis('off')

    plt.tight_layout()
    # 确保Result文件夹存在
    result_dir = 'SCUTFaiss_Pic_Search/Result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_path = os.path.join(result_dir, 'search_result_with_coords.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f'已保存检索结果拼图: {save_path}')