

# IL2A 类增量学习项目

## 一、项目简介

本项目旨在复现并改进论文《Class-Incremental Learning via Dual Augmentation》中提出的**IL2A算法**。IL2A通过类增强（classAug）和语义增强（semanAug）两种数据增强方法，解决了类增量学习中的灾难性遗忘问题。我们在此基础上进行了部分改进，以进一步提升模型性能。

## 二、实验设置

1. **数据集：** 实验使用了CIFAR-100数据集，该数据集包含100个类别，每个类别有500张训练图像和100张测试图像。为了进行类增量学习实验，将`CIFAR-100` 数据集被划分为多个不重叠的类别子集，有三种划分实验：50+**5**×10（5阶段），50+**10**×5（10阶段），40+**20**×3（20阶段）。例如，50+10×5表示第一个任务包含50个类，接下来的10个任务每个包含5个类。``
2. **基础模型**：使用`ResNet-18`模型，结合 通道注意力机制 和 空间注意力机制。
   - <u>额外尝试了`ResNet-34`和`ResNet-50`模型。</u>
3. **优化器**：使用`Adam优化器`，初始学习率为0.001，权重衰减为2e-4，学习率在每45个epoch后衰减。
   - <u>额外引入余弦退火学习率调度器，以替代原有的`StepLR`。</u>

4. **知识蒸馏**：在每个增量步骤中，通过知识蒸馏损失（来自**旧模型**的特征）保留对旧任务的知识，从而减轻遗忘效应。温度参数设置为0.1。知识蒸馏损失权重为10.0。
   - <u>额外引入动态调整知识蒸馏的温度参数和损失权重。</u>

5. **学习率调度器**：`StepLR`，每45个epoch学习率减少为原来的0.1倍。

6. **批量大小**：训练和测试时的批量大小均为64。

7. **数据处理**：对训练数据随机裁剪、随机水平翻转、颜色抖动和标准化，对测试数据标准化。
   - <u>额外增加了随机擦除。</u>

8. **数据增强**：IL2A。

9. **原型增强**：在训练过程中，保存各类别的特征原型，并使用这些原型进行额外的损失计算，帮助模型更好地保持对旧类的记忆。`protoAug`损失权重为10.0。

10. **损失函数：** 使用交叉熵损失函数、知识蒸馏损失和原型增强损失的加权和作为最终损失。

11. **训练周期：** 每个任务训练周期为10个`epoch`，每个`epoch`打印一次训练精度。

12. **注意力机制：** 局部注意力机制。
    - <u>额外引入非局部注意力机制。</u>



## 三、文件结构

- 项目目录结构如下：

```css
IL2A/
├── IL2A_50+5×10/               # 50+5×10
├── IL2A_50+10×5/               # 50+10×5
├── IL2A_40+20×3/               # 40+20×3
├── ResNet34/                   # 修改残差网络为ResNet-34
├── ResNet50/                   # 修改残差网络为ResNet-50
├── Cosine_Annealing_Scheduler/ # 加入余弦退火学习率调度器
├── Non-Local_Attention_Mechanism/ # 加入非局部注意力机制
├── Adaptive_Temperature/       # 加入自适应温度和权重算法
└── README.md                   # 项目说明文件
```

一共有9个文件夹。

| 文件夹                                   | 说明                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| IL2A_50+5×10, IL2A_50+10×5, IL2A_40+20×3 | 不同实验设置下的训练原IL2A代码。                             |
| ResNet34, ResNet50                       | 基于`ResNet-34`和`ResNet-50`模型的训练代码。                 |
| Cosine_Annealing_Scheduler               | 在`ResNet-18`和`IL2A_50+10×5`的基础实现了**余弦退火学习率调度器**，提升了模型的学习效率和稳定性。 |
| Non-Local_Attention_Mechanism            | 在余弦退火学习率调度器的基础上实现了**非局部注意力机制**，用于提升模型的全局特征提取能力。 |
| Adaptive_Temperature                     | 在余弦退火学习率调度器的基础上实现了**自适应温度和权重调整的知识蒸馏方法**，增强了模型处理新旧任务平衡的能力。 |

- 每个部分进一步目录如下（以`IL2A_50+5×10`为示例）：

```css
IL2A_50+5×10/
├── PASS.py
├── ResNet.py
├── framework.png
├── iCIFAR100.py
├── main.py
├── myNetwork.py
└── test.py
```

具体作用：

|      | 文件         | 详细功能                                                     |
| ---- | ------------ | ------------------------------------------------------------ |
| 1    | PASS.py      | 实现原型增强自监督学习（protoAugSSL）的主要功能，包括数据准备、训练、测试、模型增量学习和保存原型等功能。 |
| 2    | ResNet.py    | 使用`resnet18_cbam`作为特征提取器，并添加了通道注意力机制和空间注意力机制。 |
| 3    | iCIFAR100.py | 定义了增量学习版本的CIFAR-100数据集处理类，包括数据的获取、处理和变换等功能。 |
| 4    | main.py      | 主程序，处理参数解析，设置数据和模型，训练和测试模型，管理增量学习任务和结果输出。 |
| 5    | myNetwork.py | 定义了自定义网络类，包括前向传播、增量学习以及特征提取等功能。 |
| 6    | test.py      | 测试训练增量过程训练好的模型在每个任务上和学到的所有类上的表现。 |

## 四、使用说明

### 环境依赖

- Python 3.7+
- PyTorch 1.8+
- torchvision 0.9+
- numpy
- matplotlib

### 运行代码

1. 下载对应想要测试的文件，并进入文件夹内。

2. 训练模型：

```bash
python3 main.py
```

3. 运行测试：

```bash
python3 test.py
```

## 参考文献

1. Da-Wei Zhou, Han-Jia Ye, and De-Chuan Zhan. "Class-Incremental Learning via Dual Augmentation." In Advances in Neural Information Processing Systems (NeurIPS), 2021.
2. Zhizhong Li and Derek Hoiem. "Learning without Forgetting." In European Conference on Computer Vision (ECCV), pp. 614-629. Springer, 2016.
3. Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, and Christoph H. Lampert. "iCaRL: Incremental Classifier and Representation Learning." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2001-2010. 2017.
4. James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwińska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. "Overcoming catastrophic forgetting in neural networks." Proceedings of the National Academy of Sciences 114, no. 13 (2017): 3521-3526.
5. He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778. 2016.
6. Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the Knowledge in a Neural Network." arXiv preprint arXiv:1503.02531 (2015).
7. Kemker, Ronald, Marc McClure, Angelina Abitino, Tyler Hayes, and Christopher Kanan. "Measuring Catastrophic Forgetting in Neural Networks." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1. 2018.
8. Shin, Hanul, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. "Continual Learning with Deep Generative Replay." In Advances in Neural Information Processing Systems (NeurIPS), pp. 2990-2999. 2017.

