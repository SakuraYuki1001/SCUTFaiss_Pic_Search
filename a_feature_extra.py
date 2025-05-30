from theory import *

if __name__ == '__main__':
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