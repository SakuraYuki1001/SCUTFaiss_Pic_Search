'''
这里是定义图像特征提取处理的方法函数
'''
#导入库
import os #需要使用系统信息
from PIL import Image #PIL库用于处理图像
import torch #torch库用于深度学习
from torch.utils.data import DataLoader #导入torch.utils.data，只需要DataLoader类，Dataset我们是自定义的
from torchvision import models, transforms #从torchvision库导入已经训练好的模型，用于图像处理

#确保设备情况正确
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #检查是否有GPU可用，如果有则使用GPU，否则使用CPU
# print("【tools】正在使用的设备是:", device) #打印当前使用的设备

#按照函数处理——对图像进行预处理：将图像转换为Tensor并进行标准化
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'''
    transforms.Resize((256, 256))：将输入图像的大小缩放为256x256的尺寸。
    transforms.CenterCrop(224)：在图像的中心位置进行裁剪，将图像裁剪为224x224的尺寸。
    transforms.ToTensor()：将图像转换为PyTorch中的Tensor对象。
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])：对图像进行标准化操作，使得图像像素的数值范围符合神经网络的输入要求。
    '''

#载入模型
def load_model():
    model = models.resnet50(weights=None)  # 创建一个ResNet-50模型对象，不加载预训练权重
    model.to(device)
    model.eval()
    return model
'''
1.定义了一个名为load_model的函数，该函数不带任何参数。
2.第二行创建了一个ResNet-50模型对象，并将该模型的预训练权重加载到模型中。
3.第三行将该模型对象移动到指定的计算设备上（GPU或CPU）进行计算。
4.第四行将该模型对象设置为评估模式（即在推理时对输入数据进行前向传递）。
'''

# 定义 特征提取器
def feature_extract(model, x): #model是预训练的ResNet模型，x是输入的图像张量
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    return x
'''
model是预训练的ResNet模型，x是输入的图像张量。
图像张量x就是一个四维张量，表示批量大小、通道数、高度和宽度。
1.对输入数据张量进行第一层卷积操作，并将结果保存在x中。
2.对x进行批标准化操作，并将结果保存在x中。
3.对x进行ReLU激活操作，并将结果保存在x中。
4.对x进行最大池化操作，并将结果保存在x中。
5.对x进行四个残差块的前向传播操作，并将结果保存在x中。
6.对x进行平均池化操作，并将结果保存在x中。
# 最后将x展平为二维张量，返回提取的特征向量。
'''