# 导入依赖库
import torch
from torch import nn
from net import MyAlexNet
from net import VGG
from net import VGG_11
from net import VGG_13
from net import VGG_16
from net import VGG_19
import numpy as np
from torch.optim import lr_scheduler
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 定义训练集、验证集路径
ROOT_TRAIN = r'F:\组会\202205学术推送\SHAPC_LAB_ImgClassification\data\train'
ROOT_TEST = r'F:\组会\202205学术推送\SHAPC_LAB_ImgClassification\data\val'
# 将图像的像素值归一化到[-1， 1]之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 将训练集输入图像进行预处理，重映射尺寸、随机旋转、转化张量
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize])
# 将验证集输入图像进行预处理，重映射尺寸、随机旋转、转化张量
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])
# 设置训练集、验证集加载路径
train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
# 加载训练集、验证集
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
# 设置使用gpu进行训练，检测若无gpu，则使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将Alexnet(vgg11)结构配置到gpu(cpu)运行
model = MyAlexNet().to(device)
# model = VGG_11().to(device)
# 定义一个损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义一个优化器，将学习率设为0.0003
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 初始化总损失、总准确度、计数器
    loss, current, n = 0.0, 0.0, 0
    # 遍历获得加载进来图像的序号，数据，标签
    for batch, (x, y) in enumerate(dataloader):
        # 将数据，标签配置到gpu(cpu)中
        image, y = x.to(device), y.to(device)
        # 将数据输入Alexnet模型获得输出
        output = model(image)
        # 使用输出与真实标签计算损失
        cur_loss = loss_fn(output, y)
        # 使用输出获得预测结果
        _, pred = torch.max(output, axis=1)
        # 使用预测结果计算准确率
        cur_acc = torch.sum(y==pred) / output.shape[0]
        # 将梯度设置为0
        optimizer.zero_grad()
        # 反向传播
        cur_loss.backward()
        # 更新梯度
        optimizer.step()
        # 获得累加损失
        loss += cur_loss.item()
        # 获得累加准确率
        current += cur_acc.item()
        # 计数
        n = n+1
    # 计算训练损失
    train_loss = loss / n
    # 计算训练准确率
    train_acc = current / n
    # 在控制台打印损失及准确率
    print('train_loss' + str(train_loss))
    print('train_acc' + str(train_acc))
    # 调用该函数时，返回损失及准确率
    return train_loss, train_acc
# 定义验证函数(功能如上不再重复标注)
def val(dataloader, model, loss_fn):
    # 将模型转化为验证模型（不更新梯度）
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
    val_loss = loss / n
    val_acc = current / n
    print('val_loss' + str(val_loss))
    print('val_acc' + str(val_acc))
    return val_loss, val_acc
# 绘制损失图
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()
# 绘制准确率图
def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.show()
# 定义训练集损失、准确度空列表，定义验证集损失、准确度空列表
loss_train = []
acc_train = []
loss_val = []
acc_val = []
# 定义训练轮次
epoch = 50
# 设置初始准确度为0
min_acc = 0
# 开始训练
for t in range(epoch):
    # 显示训练进度
    print(f"epoch{t+1}\n-----------")
    # 获得训练集、验证集的损失、准确率
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)
    # 记录训练集、验证集的损失、准确率
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)
    # 保存最好的模型权重
    # 如果训练精度大于之前精度，则执行此步骤
    if val_acc >min_acc:
        folder = 'save_model'
        # 如果不存在路径则创建该路径
        if not os.path.exists(folder):
            os.mkdir('save_model')
        # 将最优准确率更新
        min_acc = val_acc
        # 打印这是第几轮出现并保存的准确率模型
        print(f"save best model, 第{t+1}轮")
        # 保存训练权重
        torch.save(model.state_dict(), 'save_model/{}best_model.pth'.format(val_acc))
    # 如果是训练最后一轮保，则执行此步骤
    if t == epoch-1:
        # 保存最后一轮模型训练权重
        torch.save(model.state_dict(), 'save_model/{}last_model.pth'.format(val_acc))
# 绘制损失和准确率图
matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print('Done!')
