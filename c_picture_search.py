from theory import *
import warnings
warnings.filterwarnings('ignore')
import faiss
import time
import numpy as np
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
加载索引库，就是从磁盘中读取之前保存的索引文件，创建一个faiss索引对象。
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



'''图片检索样例
“图片在线检索”方法函数定义——本地图片的相似度图片检索
1.本地图片->网络模型->图片特征提取->图片检索
2.加载本地索引库文件
3.检索的图片特征和本地索引库进行比对->返回相似度最高的图片
'''
# model = load_model()  #加载预训练模型

img_folder = 'SCUTFaiss_Pic_Search\PictureXyy'  # 数据集路径
img_id = '11.jpg'
topK = 20
img_path = os.path.join( img_folder,img_id)
print(img_path) # 查看  这个img_path 的相似图片

img = Image.open(img_path)
img = transform(img) # torch.Size([3, 224, 224])
img = img.unsqueeze(0) # torch.Size([1, 3, 224, 224])
img = img.to(device)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 设置可见的GPU设备，0表示使用第一个GPU
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
    print('query_img.shape = ',img_path,query_img.shape)
    # visual_plot( ind,dis,topK,query_img)