import torch.nn as nn
import torch

# 定义一个神经网络类
class network(nn.Module):
    # 初始化函数，定义网络的基本结构
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor  # 特征提取器
        self.fc = nn.Linear(512, numclass, bias=True)  # 全连接层，输入特征维度为512，输出类别数为numclass

    # 定义前向传播函数
    def forward(self, input):
        x = self.feature(input)  # 使用特征提取器提取输入数据的特征
        x = self.fc(x)  # 将特征输入到全连接层进行分类
        return x  # 返回分类结果

    # 定义增量学习函数，增加输出类别数
    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data  # 获取全连接层的权重
        bias = self.fc.bias.data  # 获取全连接层的偏置
        in_feature = self.fc.in_features  # 获取全连接层的输入特征数
        out_feature = self.fc.out_features  # 获取全连接层的输出特征数

        # 创建新的全连接层，输入特征数不变，输出特征数更新为新的类别数
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        # 保留原有权重和偏置
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

    # 定义特征提取函数，直接调用特征提取器
    def feature_extractor(self, inputs):
        return self.feature(inputs)  # 返回提取的特征
