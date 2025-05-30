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
print("正在使用的设备是:", device) #打印当前使用的设备



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
img_folder = 'SCUT_FaissSearch/SCUTFaiss_Pic_Search/PictureVOC'  # 数据集路径（一定要相对路径！）

# 创建自定义的数据集PictureVOC实例：val_dataset，将数据集文件夹路径和自定义transform预处理函数传递给MyDataset类的构造函数。
# 即可得到一个包含处理后的含有图像和相对应索引值的数据集对象。
val_dataset = MyDataset( img_folder, transform=transform)

# 创建DataLoader实例：val_dataloader，用于批量加载数据集。
batch_size = 64  # 设置批处理的图像数量
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # 创建DataLoader实例，用于批量加载数据集

print("dataset中的图片数量:", val_dataset.__len__())  # 打印数据集中的图像数量
print("迭代器:", int(val_dataset.__len__()/batch_size)+1)  
# 以上是计算并输出迭代次数，即在每个epoch中需要迭代的次数
# 其中，val_dataset.__len__()返回数据集的长度，batch_size是每个批次的图像数量，除以batch_size后加1是为了确保即使最后一个批次的图像数量不足batch_size也能被处理。

# 查看每个批batch的数据情况，包括索引和图像
batch = next(iter(val_dataloader))  
print("Batch keyspi批处理键值对:", batch.keys())  # 打印批次数据的键值对
# 获取了val_dataloader中的第一个批次数据
# 其中inter()函数是DataLoader返回的迭代器，netx()函数用于获取下一个batch数据，然后用keys()函数查看batch中的键值对。

img = batch['img']  # 获取img对应的数据
index = batch['index']  # 获取batch这个key对应的数据

print(img.shape)  # 打印出来的依次是batch的大小、通道数、图像高度和宽度
print(index)  # 打印索引值



'''
特征提取
1.对数据集中的每一张图像进行特征提取
2.将提取的特征保存到指定的文件中
3.将提取的特征保存为文本文件
（采用cpu模式或者GPU/cuda下进行图片特征提取）
'''
for idx,batch in enumerate(val_dataloader):   #索引和批次
    img = batch['img'] # 图片数据表示-> 图片特征过程？
    index = batch['index']
    
    img = img.to(device)   #将图像数据image转移到指定设备上
    feature = feature_extract( model,img )  #使用预训练模型model从图像数据img中提取图像特征feature。
    
    feature = feature.data.cpu().numpy()   #将GPU中的张量数据转移到CPU中，并转换为Numpy数组类型。
    
    imgs_path = [ os.path.join(img_folder,val_dataset.data[i] + '.txt') for i in index ]  #使用图像索引index获取图像文件路径imgs_path，其中val_dataset.data[i]表示第i个图像的文件名（不包含扩展名）。
    
    assert len(feature)== len(imgs_path) #断言特征数据feature的长度与图像文件路径imgs_path的长度相等。

    for i in range(len(imgs_path)):  
        feature_list = [ str(f) for f in feature[i] ]  #将图像特征feature转换为字符串列表feature_list，以便写入文件。
        img_path  = imgs_path[i]    #获取当前图像的文件路径img_path。
        
        with open( img_path,'w',encoding='utf-8' ) as f:   #打开文件img_path，用于写入图像特征数据。
            f.write( " ".join(feature_list) )    #将特征数据以空格分隔符连接为字符串，并写入文件。
            
    
    # print('*' * 60)  #打印分割线。
    # print(idx*batch_size)  #打印当前批次数据的起始索引（第一个数据的索引）。



'''
获取图片特征
'''
def img2feat(pic_file):  #文件路径
    feat = []    #建立空特征
    with open( pic_file ,'r',encoding='utf-8') as f:  #打开文件夹，使用utf-8的方式
        lines = f.readlines()    #读取文件的所有行，并将其存储在lines中
        feat = [float(f) for f in lines[0].split() ]  #从lines列表的第一行中提取特征向量值，并使用split()方法将其拆分为一个由浮点数字符串组成的列表。然后，使用列表推导式将这些字符串转换为浮点数，并将其存储在feat列表中。
    return feat    #返回获得的特征
    
#测试函数是否成功
'''
img_folder = 'SCUT_FaissSearch/SCUTFaiss_Pic_Search/PictureVOC'
img_path = os.path.join(img_folder, '0.jpg.txt')  # 拼接文件路径
print(img_path)
feat = img2feat(img_path)
print(feat)
'''



'''
构建图片特征索引——设置数据格式
1.基于图片特征建立索引，便于快速检索
2.这里我们使用Faiss开源向量的索引库构建图片特征索引(检索速度很快)
3.索引格式<特征,索引Id>
'''

import numpy as np #使用numpy库来处理数组和矩阵运算
ids = []  #初始化一个空列表ids，用于存储图像索引
data = []  #初始化一个空列表data，用于存储图像特征

img_folder = 'SCUT_FaissSearch/SCUTFaiss_Pic_Search/PictureVOC'  # 数据集路径
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



'''
图片特征索引创建
1.还是基于以上的Faiss工具，使用其提供的索引创建方法来创建图片特征索引
2.数据集是我们的PictureVOC数据集，特征是我们提取的图片特征
'''
import faiss  #导入Faiss库，用于高效的相似性搜索和聚类
print("Faiss版本:", faiss.__version__)  #打印Faiss库的版本信息

'''特征索引构建方案
方案1-创建图片特征索引
index = faiss.index_factory(d,"IDMap,Flat")
index.add_with_ids(data,ids)

方案2-创建图片特征索引(资源有限，效果更好 )
    IDMap 支持add_with_ids 
    如果很在意，使用”PCARx,...,SQ8“ 如果保存全部原始数据的开销太大，可以用这个索引方式。包含三个部分，
1.降维
2.聚类
3.scalar 量化，每个向量编码为8bit 不支持GPU
'''
index = faiss.index_factory(d, "IDMap,PCAR16,IVF50,SQ8")   #初始化，d表示特征向量的维度，这里为512，每个图像有512个特征数
"""
IDMap：将向量 id 映射到其在列表中的位置，实现索引中向量 id 的快速检索。
PCAR16：对输入的特征向量进行 PCA 降维，将其转换为长度为 16 的向量。这样可以减小内存占用和搜索时间，同时保留大部分特征向量的信息。
IVF50：将向量空间划分为 50 个子空间（也称为“细分”），在每个子空间上建立一个倒排表（即 IVF），以存储在该子空间中的向量。这种方法可用于加速向量检索，特别是在高维空间中。
SQ8：使用乘法量化器（SQ）将特征向量的每个分量量化为 8 个值中的一个。这可以大大减小内存占用，并加快相似度搜索。
"""
index.train(data)   #对数据集进行训练，以构建索引。
index.add_with_ids(data, ids)   #将数据集和ID添加到索引中。

# 索引文件保存磁盘
faiss.write_index(index,'index_file.index') # 其中，faiss是用于高效相似度搜索和聚类的库，write_index是该库中用于将索引对象写入磁盘文件的方法。具体地，这里将索引对象index写入了名为index_file.index的文件。
index = faiss.read_index("index_file.index")#这行代码的作用是从磁盘中读取索引文件index_file.index并创建一个faiss索引对象。该对象可用于对数据集进行快速搜索。
print(index.ntotal) # 查看索引库大小



'''
实现图片检索功能
'''
# 确保需要的库的导入
import warnings
warnings.filterwarnings('ignore')
import faiss
import os
import time
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
'''
warnings：用于控制警告信息的输出。
faiss：Facebook AI Research开源的高效相似度搜索库，用于向量相似度搜索。
os：用于操作系统相关的功能，比如文件读写等。
time：用于时间相关的功能，比如计算代码运行时间等。
numpy：用于科学计算和数值运算。
torch：PyTorch深度学习框架。
torchvision：PyTorch的视觉工具库，包括图像处理、图像分类等功能。
transforms：PyTorch中用于数据增强的类，比如图像缩放、翻转、裁剪等。
DataLoader：PyTorch中用于批量读取数据的类。
PIL：Python图像处理库。
matplotlib：Python可视化库，用于画图。
    import matplotlib.pyplot as plt
    img = Image.open('your_image.jpg')
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴（可选）
    plt.show()       # 显示图像
'''

'''
加载索引
'''
index = faiss.read_index("index_file.index")  # 从磁盘中读取索引文件，创建一个faiss索引对象
print("索引库大小:", index.ntotal)  # 打印索引库的大小，即索引中存储的向量数量



'''
定义检索方法
'''
def index_search(feat,topK ):
    '''
        feat: 检索的图片特征
        topK: 返回最高topK相似的图片
    '''

    feat = np.expand_dims( np.array(feat),axis=0 )    #将feat数组进行扩展，增加一维
    feat = feat.astype('float32')    #数据类型转换为float32
    
    start_time = time.time()
    dis,ind = index.search( feat,topK )  #递归调用
    end_time = time.time()
    
    print( 'index_search consume time:{}ms'.format(  int(end_time - start_time) * 1000  ) )   #打印查询花费的时间
    return dis,ind # 距离，相似图片id



'''
“图片可视化”方法函数定义
'''
def visual_plot(ind,dis,topK,query_img = None):  
    """
    ind：搜索结果的索引
dis：搜索结果的距离
topK：返回的最相似的前k个结果
query_img（可选）：作为查询的图像，用于在可视化中显示
    """
    # 相似照片
    cols = 4   #每一行显示的照片数  
    rows = int(topK / cols)  #总共需要的行数
    idx = 0   #idx初始化为0

    fig,axes = plt.subplots(rows,cols,figsize=(20 ,5*rows),tight_layout=True)
    """
    rows 和 cols 是子图的行数和列数，分别通过 topK 变量计算得出。
    figsize 是整个图形对象的大小，设置为 (20, 5*rows)，表示宽度为 20，高度为每行 5 个子图的高度总和。
    tight_layout 表示是否自动调整子图之间的间距，使得整个图形更美观。
    """
    #axes[0,0].imshow(query_img)

    #axes[0,0].imshow(query_img)
    
    for row in range(rows):
        for col in range(cols):
            _id = ind[0][idx]
            _dis = dis[0][idx]
            
            img_folder = '/home/xerfia/SCUT_FaissSearch/SCUTFaiss_Pic_Search/PictureVOC'  # 数据集路径
            img_path = os.path.join(img_folder,'{}.jpg'.format(_id))
            #print(img_path)

            if query_img is not None and idx == 0:
                axes[row,col].imshow(query_img)
                axes[row,col].set_title( 'query',fontsize = 20  )
            else:
                img = plt.imread(  img_path   )
                axes[row,col].imshow(img)
                axes[row,col].set_title( 'matched_-{}_{}'.format(_id,int(_dis)) ,fontsize = 20  )
            idx+=1

    plt.savefig('pic')



'''
“图片在线检索”方法函数定义——本地图片的相似度图片检索
1.本地图片->网络模型->图片特征提取->图片检索
2.加载本地索引库文件
3.检索的图片特征和本地索引库进行比对->返回相似度最高的图片
'''
model = load_model()  #加载预训练模型

img_folder = '/home/xerfia/SCUT_FaissSearch/SCUTFaiss_Pic_Search/PictureVOC'  # 数据集路径
img_id = '11.jpg'
topK = 20
img_path = os.path.join( img_folder,img_id)
print(img_path) # 查看  这个img_path 的相似图片

img = Image.open(img_path)
img = transform(img) # torch.Size([3, 224, 224])
img = img.unsqueeze(0) # torch.Size([1, 3, 224, 224])
img = img.to(device)

# 对我们的图片进行预测
with torch.no_grad():
    # 图片-> 图片特征
    print('1.图片特征提取')
    feature = feature_extract( model,img )
    # 特征-> 检索
    feature_list = feature.data.cpu().tolist()[0]
    print('2.基于特征的检索，从faiss获取相似度图片')
    # 相似图片可视化
    dis,ind = index_search( feature_list,topK=topK )
    print('ind = ',ind)
    print('3.图片可视化展示')
    # 当前图片
    query_img = plt.imread( img_path )
    visual_plot( ind,dis,topK,query_img)

'''
互联网图片检索
'''
#!pip install scikit-image==0.16.2-------该库图像读取、变换、过滤、分割、特征提取
topK = 10  #需要返回20张图片
# 狗
img_src = '16.jpg'
from skimage import io   
img = io.imread(img_src)   #imread函数读取了一个图像文件，该函数返回一个numpy数组类型的图像
img = Image.fromarray(img)  #ndarray_image为原来的numpy数组类型的输入，将numpy数组类型的图像转换为Image对象类型，使用了Pillow库中的Image模块
query_img = img  #Image对象赋值给query_img变量

img_folder = 'VOC2012_small/'
img = transform(img) # torch.Size([3, 224, 224])  #将图像转换为大小为3x224x224的张量
img = img.unsqueeze(0) #使用了PyTorch的unsqueeze函数，将维度为1的新维度添加到张量的第一维，即将大小为[3, 224, 224]的张量变为大小为[1, 3, 224, 224]的张量
img = img.to(device)  #将张量移动到指定的设备（如CPU或GPU）上进行处理
with torch.no_grad():   #(简单说就是不计算模型的梯度，因为梯度是为了求权重的)使用no_grad函数可以临时禁用梯度计算，以加快模型的推理速度。同时，这也可以避免在推理时意外修改模型的权重。使用no_grad函数可以临时禁用梯度计算，以加快模型的推理速度。同时，这也可以避免在推理时意外修改模型的权重。
    print('1.*****************')
    # 图片->提取特征
    feature = feature_extract(model, img)
    print('2.*****************')
    feature_list = feature.data.cpu().tolist()[0]  #使用了cpu函数将其移动到CPU设备上。然后，使用了tolist函数将特征转换为Python列表格式，并取出其中的第一个元素。由于特征通常是一个大小为[1, n]的张量，因此使用[0]索引可以获取特征列表中的唯一元素，即大小为[n]的一维列表。
    # 特征-> 检索
    print('3.*****************' )
    #print(feature_list)
    dis,ind = index_search(feature_list,topK=topK)
    #print(dis)
    print(ind)#打印最相似的索引
    print('4.****************')
    visual_plot(ind,dis,topK,query_img)