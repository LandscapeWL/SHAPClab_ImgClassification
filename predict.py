import torch
from torch import nn
from net import MyAlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
from PIL import Image
import os
import csv
# 设置预测图像本地路径
imagepath = r'F:\组会\202205学术推送\SHAPC_LAB_ImgClassification\predict_img/'
# 将预测输入图像进行预处理，重映射尺寸、随机旋转、转化张量、归一化
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# 设置使用gpu进行训练，检测若无gpu，则使用cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 加载模型，并将模型配置到gpu(cpu)中
model = MyAlexNet().to(device)
# 使用gpu
model.cuda()
# 载入训练的模型权重
model.load_state_dict(torch.load(r"F:\组会\202205学术推送\SHAPC_LAB_ImgClassification\save_model/0.7787698422159467best_model.pth"))
# 定义预测结果分类名称
# classes = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]
classes = ["agricultural","airplane","baseballdiamond","beach","buildings","chaparral","denseresidential","forest","freeway",
           "golfcourse","harbor","intersection","mediumresidential","mobilehomepark","overpass","parkinglot","river","runway",
           "sparseresidential","storagetanks","tenniscourt"]
# 创建predict.csv文件用以保存预测结果
with open('predict.csv', 'w',newline='')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(['序号','结果'])
    # 获得预测所有图像文件名
    imgs = os.listdir(imagepath)
    # 遍历文件名，逐个预测处理
    for jpg in imgs:
        # 打开单个图像文件，并进行预处理
        x = Image.open(imagepath + jpg)
        x = data_transforms(x)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
        x = torch.tensor(x).to(device)
        # 不进行梯度更新
        with torch.no_grad():
            # 将图像数据输入模型，输出预测结果
            pred = model(x)
            predicted = classes[torch.argmax(pred[0])]
            print('图片名:' + jpg, ',', '预测结果：', predicted)
        # 将结果写入csv
        f_csv.writerow([jpg.split('_')[0],predicted])




