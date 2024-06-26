import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import sys
import numpy as np
from myNetwork import network
from iCIFAR100 import iCIFAR100


# 定义原型增强自监督学习（protoAugSSL）类
class protoAugSSL:
    # 初始化函数，定义类的基本属性和参数
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name  # 文件名
        self.args = args  # 命令行参数
        self.epochs = args.epochs  # 训练周期数
        self.learning_rate = args.learning_rate  # 学习率
        self.model = network(args.fg_nc * 4, feature_extractor)  # 初始化模型，类别数乘以4
        self.radius = 0  # 半径，用于原型增强
        self.prototype = None  # 原型存储
        self.class_label = None  # 类别标签
        self.numclass = args.fg_nc  # 类别数
        self.task_size = task_size  # 每个任务的类别数
        self.device = device  # 设备（GPU或CPU）
        self.old_model = None  # 用于知识蒸馏的旧模型

        # 定义训练数据的变换
        self.train_transform = transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.ColorJitter(brightness=0.24705882352941178),  # 随机颜色抖动
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 标准化
        ])

        # 定义测试数据的变换
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # 标准化
        ])

        # 初始化训练和测试数据集
        self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None  # 训练数据加载器
        self.test_loader = None  # 测试数据加载器

    # 映射新类别索引
    def map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    # 设置数据集的类别顺序
    def setup_data(self, shuffle, seed):
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()  # 随机打乱顺序
        else:
            order = range(len(order))
        self.class_order = order  # 类别顺序
        print(100 * '#')
        print(self.class_order)

        # 映射训练和测试数据集的类别索引
        self.train_dataset.targets = self.map_new_class_index(train_targets, self.class_order)
        self.test_dataset.targets = self.map_new_class_index(test_targets, self.class_order)

    # 训练前的准备工作
    def beforeTrain(self, current_task):
        self.model.eval()  # 将模型设置为评估模式
        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass - self.task_size, self.numclass]
        # 获取训练和测试数据加载器
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(4 * self.numclass)  # 增加输出类别数
        self.model.train()  # 将模型设置为训练模式
        self.model.to(self.device)  # 将模型移动到指定设备

    # 获取训练和测试数据加载器
    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)  # 获取训练数据
        self.test_dataset.getTestData(classes)  # 获取测试数据
        train_loader = DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.args.batch_size)
        test_loader = DataLoader(dataset=self.test_dataset, shuffle=True, batch_size=self.args.batch_size)
        return train_loader, test_loader

    # 获取测试数据加载器
    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset, shuffle=True, batch_size=self.args.batch_size)
        return test_loader

    # 训练函数
    def train(self, current_task, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)  # 学习率调度器
        accuracy = 0
        for epoch in range(self.epochs):
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                # 自监督学习的标签增强
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                opt.zero_grad()  # 清零梯度
                loss = self._compute_loss(images, target, old_class)  # 计算损失
                opt.zero_grad()
                loss.backward()  # 反向传播
                opt.step()  # 优化更新
            scheduler.step()  # 更新学习率
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)  # 测试模型
                print('训练周期(epoch):%d, 准确率(accuracy):%.5f' % (epoch, accuracy))
        self.protoSave(self.model, self.train_loader, current_task)  # 保存原型

    # 测试函数
    def _test(self, testloader):
        self.model.eval()  # 设置模型为评估模式
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            outputs = outputs[:, ::4]  # 仅计算原始类别节点的预测
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()  # 设置模型为训练模式
        return accuracy

    # 计算知识蒸馏损失函数
    # 在每一个增量步骤中，当前模型在学习新任务的同时，通过知识蒸馏损失（来自旧模型的特征）保留对旧任务的知识，从而减轻遗忘效应
    def _compute_loss(self, imgs, target, old_class=0):
        output = self.model(imgs)
        output, target = output.to(self.device), target.to(self.device)
        loss_cls = nn.CrossEntropyLoss()(output / self.args.temp, target)  # 计算当前模型的分类损失
        if self.old_model is None:
            return loss_cls
        else:  # 如果旧模型存在，则计算当前模型和旧模型的特征之间的距离，作为知识蒸馏损失
            feature = self.model.feature(imgs)
            feature_old = self.old_model.feature(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)  # 知识蒸馏损失

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            for _ in range(self.args.batch_size):
                np.random.shuffle(index)
                temp = self.prototype[index[0]] + np.random.normal(0, 1, 512) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(4 * self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.fc(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / self.args.temp, proto_aug_label)  # 原型增强损失

            # 最终的损失是分类损失、原型增强损失和知识蒸馏损失的加权和
            return loss_cls + self.args.protoAug_weight * loss_protoAug + self.args.kd_weight * loss_kd

    # 训练后的处理工作
    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'  # 构建模型保存路径
        if not os.path.isdir(path):  # 如果路径不存在，创建该路径
            os.makedirs(path)
        self.numclass += self.task_size  # 更新类别数量，增加一个任务的类别数
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)  # 根据当前任务的类别数生成文件名
        torch.save(self.model, filename)  # 保存当前模型
        self.old_model = torch.load(filename)  # 加载旧模型
        self.old_model.to(self.device)
        self.old_model.eval()  # 将旧模型设置为评估模式，以便在下一个增量学习任务中使用

    # 保存原型
    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(loader):
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
