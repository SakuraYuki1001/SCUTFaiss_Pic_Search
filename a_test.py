'''
获取图片特征
'''
from a_feature_extra import *  #从a_feature_extra模块导入所有内容
def img2feat(pic_file):  #文件路径
    feat = []    #建立空特征
    with open( pic_file ,'r',encoding='utf-8') as f:  #打开文件夹，使用utf-8的方式
        lines = f.readlines()    #读取文件的所有行，并将其存储在lines中
        feat = [float(f) for f in lines[0].split() ]  #从lines列表的第一行中提取特征向量值，并使用split()方法将其拆分为一个由浮点数字符串组成的列表。然后，使用列表推导式将这些字符串转换为浮点数，并将其存储在feat列表中。
    return feat    #返回获得的特征
    
#测试函数是否成功
'''
img_path = os.path.join(img_folder, 't01a6e2e9dc76400120.jpg.txt')  # 拼接文件路径
print(img_path)
feat = img2feat(img_path)
print(feat)
'''