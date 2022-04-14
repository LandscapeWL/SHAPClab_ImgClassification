import os
from shutil import copy
import random
# 定义类：作用是检查file路径是否存在，若不存在则创建该路径
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
# 定义数据集的路径，请切换为你电脑中所使用的路径
file_path = r'F:\组会\202205学术推送\SHAPC_LAB_ImgClassification\Images'
# 获取file_path路径下所有文件名（即需要分类的类名）
flower_class = [cla for cla in os.listdir(file_path)]
# 创建 训练集train文件夹，并由类名在其目录下创建每个子目录
mkfile('data/train')
for cla in flower_class:
    mkfile('data/train/' + cla)
# 创建 验证集val文件夹，并由类名在其目录下创建每个子目录
mkfile('data/val')
for cla in flower_class:
    mkfile('data/val/' + cla)
# 数据划分比例，训练集:验证集=9:1
split_rate = 0.1
# 遍历所有类别的全部图像并按比例分成训练集和验证集
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录路径
    images = os.listdir(cla_path)  # iamges 存储了该目录下所有图像的名称
    num = len(images) # 使用len方法计算images中的图片个数
    eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k个图像
    for index, image in enumerate(images):
        # eval_index 中保存验证集val的图像名称
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径
        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        # 事实打印显示处理进度
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
    print()
# 处理完成，打印“processing done！”
print("processing done!")
