'''
图片特征索引创建
1.还是基于以上的Faiss工具，使用其提供的索引创建方法来创建图片特征索引
2.数据集是我们的PictureVOC数据集，特征是我们提取的图片特征
'''
from b_feature_index_structure import *
'''
图片特征索引创建
1.还是基于以上的Faiss工具，使用其提供的索引创建方法来创建图片特征索引
2.数据集是我们的PictureVOC数据集，特征是我们提取的图片特征
'''
import faiss  #导入Faiss库，用于高效的相似性搜索和聚类
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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