import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

# 定义模块名称，可以导入模块时直接使用这些名称
__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam', 'resnet152_cbam']

# 预训练模型的URL地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 定义一个3x3卷积层，包含padding
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# 定义通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)  # 第一个全连接层，使用1x1卷积实现
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)  # 第二个全连接层

        self.sigmoid = nn.Sigmoid()  # 使用sigmoid激活函数

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化通道注意力
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化通道注意力
        out = avg_out + max_out  # 相加融合
        return self.sigmoid(out)  # 激活并输出

# 定义空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 定义卷积层
        self.sigmoid = nn.Sigmoid()  # 使用sigmoid激活函数

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 在通道维度上拼接
        x = self.conv1(x)  # 卷积
        return self.sigmoid(x)  # 激活并输出

# 定义基本块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活
        self.conv2 = conv3x3(planes, planes)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(planes)  # 批归一化

        self.ca = ChannelAttention(planes)  # 通道注意力机制
        self.sa = SpatialAttention()  # 空间注意力机制

        self.downsample = downsample  # 下采样层
        self.stride = stride  # 步幅

    def forward(self, x):
        residual = x  # 残差连接
        out = self.conv1(x)  # 第一个卷积
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # ReLU激活
        out = self.conv2(out)  # 第二个卷积
        out = self.bn2(out)  # 批归一化

        # 应用通道注意力机制
        out = self.ca(out) * out
        # 应用空间注意力机制
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)  # 下采样

        out += residual  # 残差连接
        out = self.relu(out)  # ReLU激活
        return out  # 输出

# 定义瓶颈块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # 1x1卷积层
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # 3x3卷积层
        self.bn2 = nn.BatchNorm2d(planes)  # 批归一化
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  # 1x1卷积层
        self.bn3 = nn.BatchNorm2d(planes * 4)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活

        self.ca = ChannelAttention(planes * 4)  # 通道注意力机制
        self.sa = SpatialAttention()  # 空间注意力机制

        self.downsample = downsample  # 下采样层
        self.stride = stride  # 步幅

    def forward(self, x):
        residual = x  # 残差连接
        out = self.conv1(x)  # 1x1卷积
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # ReLU激活
        out = self.conv2(out)  # 3x3卷积
        out = self.bn2(out)  # 批归一化
        out = self.relu(out)  # ReLU激活
        out = self.conv3(out)  # 1x1卷积
        out = self.bn3(out)  # 批归一化

        # 应用通道注意力机制
        out = self.ca(out) * out
        # 应用空间注意力机制
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)  # 下采样

        out += residual  # 残差连接
        out = self.relu(out)  # ReLU激活
        return out  # 输出

# 定义ResNet类
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 第一层卷积
        self.bn1 = nn.BatchNorm2d(64)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层
        self.layer1 = self._make_layer(block, 64, layers[0])  # 第一层残差块
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 第二层残差块
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 第三层残差块
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 第四层残差块
        self.feature = nn.AvgPool2d(4, stride=1)  # 平均池化层
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 创建残差块
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 第一个残差块
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # 其他残差块

        return nn.Sequential(*layers)  # 返回顺序容器

    # 定义前向传播
    def forward(self, x):
        x = self.conv1(x)  # 卷积
        x = self.bn1(x)  # 批归一化
        x = self.relu(x)  # ReLU激活
        x = self.layer1(x)  # 第一层残差块
        x = self.layer2(x)  # 第二层残差块
        x = self.layer3(x)  # 第三层残差块
        x = self.layer4(x)  # 第四层残差块
        dim = x.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)  # 平均池化
        x = pool(x)
        x = x.view(x.size(0), -1)  # 展平
        return x  # 返回特征

# 定义ResNet-18模型
def resnet18_cbam(pretrained=False, **kwargs):
    """构建ResNet-18模型。
    参数:
        pretrained (bool): 如果为True，则返回一个在ImageNet上预训练的模型
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)  # 使用BasicBlock构建ResNet-18，包含4个层，每层包含2个BasicBlock
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])  # 加载预训练模型参数
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)  # 更新模型参数
        model.load_state_dict(now_state_dict)  # 将更新后的参数加载到模型中
    return model  # 返回构建好的模型

# 定义ResNet-34模型
def resnet34_cbam(pretrained=False, **kwargs):
    """构建ResNet-34模型。
    参数:
        pretrained (bool): 如果为True，则返回一个在ImageNet上预训练的模型
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)  # 使用BasicBlock构建ResNet-34，包含4个层，每层包含3、4、6、3个BasicBlock
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])  # 加载预训练模型参数
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)  # 更新模型参数
        model.load_state_dict(now_state_dict)  # 将更新后的参数加载到模型中
    return model  # 返回构建好的模型

# 定义ResNet-50模型
def resnet50_cbam(pretrained=False, **kwargs):
    """构建ResNet-50模型。
    参数:
        pretrained (bool): 如果为True，则返回一个在ImageNet上预训练的模型
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)  # 使用Bottleneck构建ResNet-50，包含4个层，每层包含3、4、6、3个Bottleneck
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])  # 加载预训练模型参数
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)  # 更新模型参数
        model.load_state_dict(now_state_dict)  # 将更新后的参数加载到模型中
    return model  # 返回构建好的模型

# 定义ResNet-101模型
def resnet101_cbam(pretrained=False, **kwargs):
    """构建ResNet-101模型。
    参数:
        pretrained (bool): 如果为True，则返回一个在ImageNet上预训练的模型
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)  # 使用Bottleneck构建ResNet-101，包含4个层，每层包含3、4、23、3个Bottleneck
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])  # 加载预训练模型参数
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)  # 更新模型参数
        model.load_state_dict(now_state_dict)  # 将更新后的参数加载到模型中
    return model  # 返回构建好的模型

# 定义ResNet-152模型
def resnet152_cbam(pretrained=False, **kwargs):
    """构建ResNet-152模型。
    参数:
        pretrained (bool): 如果为True，则返回一个在ImageNet上预训练的模型
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)  # 使用Bottleneck构建ResNet-152，包含4个层，每层包含3、8、36、3个Bottleneck
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])  # 加载预训练模型参数
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)  # 更新模型参数
        model.load_state_dict(now_state_dict)  # 将更新后的参数加载到模型中
    return model  # 返回构建好的模型
