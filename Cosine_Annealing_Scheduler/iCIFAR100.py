from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# 定义增量学习版本的CIFAR-100数据集处理类
class iCIFAR100(CIFAR100):
    # 初始化函数，定义一些必要的参数和属性
    def __init__(self, root, train=True, transform=None, target_transform=None, test_transform=None, target_test_transform=None, download=False):
        # 调用父类CIFAR100的初始化函数
        super(iCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.target_test_transform = target_test_transform  # 测试数据的目标变换函数
        self.test_transform = test_transform  # 测试数据的变换函数
        self.TrainData = np.empty((0, 32, 32, 3), dtype=np.uint8)  # 根据CIFAR100的图片尺寸初始化
        self.TrainLabels = np.empty((0,), dtype=int)  # 初始化训练标签列表
        self.TestData = np.empty((0, 32, 32, 3), dtype=np.uint8)  # 根据CIFAR100的图片尺寸初始化
        self.TestLabels = np.empty((0,), dtype=int)  # 测试标签列表

    # 合并多个数据集和标签集
    def concatenate(self, datas, labels):
        # 初始化合并后的数据和标签
        con_data = datas[0]
        con_label = labels[0]
        # 遍历所有的数据和标签，进行拼接
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        # 返回拼接后的数据和标签
        return con_data, con_label

    # 获取指定类别范围的测试数据
    def getTestData(self, classes):
        # 初始化数据和标签列表
        datas, labels = [], []
        # 遍历指定范围内的每一个类别
        for label in range(classes[0], classes[1]):
            # 获取当前类别的所有数据
            data = self.data[np.array(self.targets) == label]
            # 将数据添加到数据列表中
            datas.append(data)
            # 生成当前类别的标签，并添加到标签列表中
            labels.append(np.full((data.shape[0]), label))
        # 合并所有获取到的数据和标签
        datas, labels = self.concatenate(datas, labels)
        # 更新测试数据和标签
        self.TestData = np.concatenate((self.TestData, datas), axis=0) if self.TestData.size else datas
        self.TestLabels = np.concatenate((self.TestLabels, labels), axis=0) if self.TestLabels.size else labels
        # 打印测试数据和标签的大小
        print("测试集大小 %s" % (str(self.TestData.shape)))
        print("测试标签大小 %s" % str(self.TestLabels.shape))

    # 获取指定类别范围的当前及之前所有类别的测试数据
    def getTestData_up2now(self, classes):
        # 初始化数据和标签列表
        datas, labels = [], []
        # 遍历指定范围内的每一个类别
        for label in range(classes[0], classes[1]):
            # 获取当前类别的所有数据
            data = self.data[np.array(self.targets) == label]
            # 将数据添加到数据列表中
            datas.append(data)
            # 生成当前类别的标签，并添加到标签列表中
            labels.append(np.full((data.shape[0]), label))
        # 合并所有获取到的数据和标签
        datas, labels = self.concatenate(datas, labels)
        # 更新测试数据和标签
        self.TestData = datas
        self.TestLabels = labels
        # 打印测试数据和标签的大小
        print("测试集大小： %s" % (str(datas.shape)))
        print("测试标签大小： %s" % str(labels.shape))

    # 获取指定类别范围的训练数据
    def getTrainData(self, classes):
        # 初始化数据和标签列表
        datas, labels = [], []
        # 遍历指定范围内的每一个类别
        for label in range(classes[0], classes[1]):
            # 获取当前类别的所有数据
            data = self.data[np.array(self.targets) == label]
            # 将数据添加到数据列表中
            datas.append(data)
            # 生成当前类别的标签，并添加到标签列表中
            labels.append(np.full((data.shape[0]), label))
        # 合并所有获取到的数据和标签
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        # 打印训练数据和标签的大小
        print("训练集大小： %s" % (str(self.TrainData.shape)))
        print("训练标签大小： %s" % str(self.TrainLabels.shape))

    # 获取训练数据集中的一个数据项
    def getTrainItem(self, index):
        # 从训练数据中获取指定索引的数据和标签
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        # 如果定义了数据变换函数，则对图像进行变换
        if self.transform:
            img = self.transform(img)
        # 如果定义了目标变换函数，则对标签进行变换
        if self.target_transform:
            target = self.target_transform(target)
        # 返回索引、图像和标签
        return index, img, target

    # 获取测试数据集中的一个数据项
    def getTestItem(self, index):
        # 从测试数据中获取指定索引的数据和标签
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        # 如果定义了测试数据变换函数，则对图像进行变换
        if self.test_transform:
            img = self.test_transform(img)
        # 如果定义了测试目标变换函数，则对标签进行变换
        if self.target_test_transform:
            target = self.target_test_transform(target)
        # 返回索引、图像和标签
        return index, img, target

    # 根据索引获取数据项
    def __getitem__(self, index):
        # 如果训练数据不为空，则获取训练数据项
        if self.TrainData.size > 0:
            return self.getTrainItem(index)
        # 否则，获取测试数据项
        elif self.TestData.size > 0:
            return self.getTestItem(index)
        else:
            raise IndexError("Index out of range")

    # 获取数据集的长度
    def __len__(self):
        # 返回训练数据或测试数据的长度
        if self.TrainData.size > 0:
            return len(self.TrainData)
        elif self.TestData.size > 0:
            return len(self.TestData)
        else:
            return 0

    # 获取某一类别的图像
    def get_image_class(self, label):
        # 返回指定类别的所有图像
        return self.data[np.array(self.targets) == label]
