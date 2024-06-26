import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import math
import argparse
import resource
import pandas as pd
import numpy as np
import sklearn.metrics
from scipy import stats
from PIL import Image

from PASS import protoAugSSL
from ResNet import resnet18_cbam
from myNetwork import network
from iCIFAR100 import iCIFAR100

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=101, type=int, help='总训练周期数')
parser.add_argument('--batch_size', default=64, type=int, help='训练批次大小')
parser.add_argument('--print_freq', default=10, type=int, help='打印频率（默认：10）')
parser.add_argument('--data_name', default='cifar100', type=str, help='使用的数据集名称')
parser.add_argument('--total_nc', default=100, type=int, help='数据集中的总类别数')
parser.add_argument('--fg_nc', default=40, type=int, help='第一个任务中的类别数')
parser.add_argument('--task_num', default=20, type=int, help='增量步骤的数量')
parser.add_argument('--learning_rate', default=0.001, type=float, help='初始学习率')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug损失权重')
parser.add_argument('--kd_weight', default=10.0, type=float, help='知识蒸馏损失权重')
parser.add_argument('--temp', default=0.1, type=float, help='训练时的温度参数')
parser.add_argument('--gpu', default='0', type=str, help='使用的GPU ID')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='模型保存目录')

# 解析命令行参数
args = parser.parse_args()
print(args)


# 映射新类别索引
def map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


# 设置数据集的类别顺序
def setup_data(test_targets, shuffle, seed):
    # 获取所有类别的顺序
    order = [i for i in range(len(np.unique(test_targets)))]
    if shuffle:
        np.random.seed(seed)
        order = np.random.permutation(len(order)).tolist()  # 随机打乱顺序
    else:
        order = range(len(order))
    class_order = order
    print(100 * '#')
    print(class_order)
    return map_new_class_index(test_targets, class_order)


# 主函数
def main():
    # 设置设备，优先使用GPU
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    # 计算每个增量步骤中的类别数量
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size)
    feature_extractor = resnet18_cbam()

    # 初始化模型
    model = protoAugSSL(args, file_name, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))
    model.setup_data(shuffle=True, seed=1993)

    # 逐个增量步骤进行训练
    for i in range(args.task_num + 1):
        if i == 0:  # 初始的类学习
            old_class = 0
        else:  # 每次增加固定类
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        model.beforeTrain(i)  # 训练前的准备工作
        model.train(i, old_class=old_class)
        # 这里会输出'训练周期(epoch):%d, 准确率(accuracy):%.5f'
        model.afterTrain()  # 训练后的处理工作，保存当前模型，并将其加载为旧模型，以便在接下来的增量步骤中使用


    ####### 测试，这里开始用到测试集 ######
    # 测试数据的预处理步骤，包括将图像转换为张量并进行归一化处理
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    print("############# 每个模型单独对每个任务的数据测试 #############")
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    test_dataset.targets = setup_data(test_dataset.targets, shuffle=True, seed=1993)
    acc_all = []

    # 计算类别范围
    category_ranges = []
    for i in range(args.task_num + 1):
        start_class = args.fg_nc if i == 0 else args.fg_nc + (i - 1) * task_size
        end_class = args.fg_nc + i * task_size
        category_ranges.append(f"[{start_class}-{end_class - 1}]")
    # 输出类别范围
    print("测试类别范围：" + ', '.join(category_ranges))

    # 对每个增量步骤进行测试（每个模型）
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        acc_up2now = []

        # 测试当前及之前的所有任务
        for i in range(current_task + 1):
            if i == 0:
                classes = [0, args.fg_nc]
            else:
                classes = [args.fg_nc + (i - 1) * task_size, args.fg_nc + i * task_size]
            test_dataset.getTestData_up2now(classes)
            test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=args.batch_size)
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(imgs)
                outputs = outputs[:, ::4]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)

        # 如果当前任务数 current_task 少于总任务数 args.task_num，用零填补未完成任务的准确率。
        if current_task < args.task_num:  # 只要不是最后一个任务，都要做这步
            acc_up2now.extend((args.task_num - current_task) * [0])
        acc_all.append(acc_up2now)
        print(acc_up2now)
    print("所有模型每个任务表现总结：")
    print(acc_all)

    print("############# 每个模型对所有之前任务数据测试 #############")
    test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    test_dataset.targets = setup_data(test_dataset.targets, shuffle=True, seed=1993)

    # 测试至今所有任务的模型
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=args.batch_size)
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            outputs = outputs[:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print(accuracy)


# 判断是否为主程序入口
if __name__ == "__main__":
    main()
