import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import numpy as np
from iCIFAR100 import iCIFAR100
import pandas as pd

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Test Trained Models')
parser.add_argument('--epochs', default=101, type=int, help='总训练周期数')
parser.add_argument('--batch_size', default=64, type=int, help='训练批次大小')
parser.add_argument('--print_freq', default=10, type=int, help='打印频率（默认：10）')
parser.add_argument('--data_name', default='cifar100', type=str, help='使用的数据集名称')
parser.add_argument('--total_nc', default=100, type=int, help='数据集中的总类别数')
parser.add_argument('--fg_nc', default=50, type=int, help='第一个任务中的类别数')
parser.add_argument('--task_num', default=10, type=int, help='增量步骤的数量')
parser.add_argument('--learning_rate', default=0.001, type=float, help='初始学习率')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug损失权重')
parser.add_argument('--kd_weight', default=10.0, type=float, help='知识蒸馏损失权重')
parser.add_argument('--temp', default=0.1, type=float, help='训练时的温度参数')
parser.add_argument('--gpu', default='0', type=str, help='使用的GPU ID')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='模型保存目录')
args = parser.parse_args()

# 设置设备，优先使用GPU
cuda_index = 'cuda:' + args.gpu
device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")

# 数据变换
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# 准备测试数据集
test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)

# 计算每个增量步骤中的类别数量
task_size = int((args.total_nc - args.fg_nc) / args.task_num)
file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size)


# 映射新类别索引
def map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


# 设置数据集的类别顺序
def setup_data(test_targets, shuffle, seed):
    order = [i for i in range(len(np.unique(test_targets)))]
    if shuffle:
        np.random.seed(seed)
        order = np.random.permutation(len(order)).tolist()
    else:
        order = range(len(order))
    class_order = order
    print(100 * '#')
    print(class_order)
    return map_new_class_index(test_targets, class_order)


# 设置数据集的类别顺序
test_dataset.targets = setup_data(test_dataset.targets, shuffle=True, seed=1993)

# 记录所有模型的准确率
acc_all = []

print("############# 每个模型单独对每个任务的数据测试 #############")

# 逐个测试模型
for current_task in range(args.task_num + 1):
    class_index = args.fg_nc + current_task * task_size
    filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
    model = torch.load(filename)
    model.to(device)
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
        for _, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            outputs = outputs[:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        acc_up2now.append(accuracy)

    # 填补未完成任务的准确率
    if current_task < args.task_num:
        acc_up2now.extend((args.task_num - current_task) * [0])
    acc_all.append(acc_up2now)
    print(acc_up2now)

# 总结
# 计算类别范围
category_ranges = []
for i in range(args.task_num + 1):
    if i == 0:
        start_class = 0
        end_class = args.fg_nc
    else:
        start_class = args.fg_nc + (i - 1) * task_size
        end_class = args.fg_nc + i * task_size
    category_ranges.append(f"[{start_class}-{end_class - 1}]")
# 创建 DataFrame 并设置行列索引
acc_df = pd.DataFrame(acc_all, index=[f"Model {i}" for i in range(args.task_num + 1)], columns=category_ranges)
print("**所有模型每个任务表现总结：**")
print(acc_df)


# 逐个测试每个模型对所有之前任务数据
print("############# 每个模型对所有之前任务数据测试 #############")
# 记录每个模型对所有之前任务数据的准确率
acc_summary = []

# 逐个测试每个模型对所有之前任务数据
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
    for _, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
        outputs = outputs[:, ::4]
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = correct.item() / total
    acc_summary.append(accuracy)
    print(accuracy)

# 使用 DataFrame 格式化输出
acc_summary_df = pd.DataFrame(acc_summary, columns=['Accuracy'], index=[f'Model {i}' for i in range(args.task_num + 1)])
print("**所有模型学到的任务表现总结：**")
print(acc_summary_df)

print(100 * '#')
print("**所有模型每个任务表现总结：**")
acc_df = acc_df.round(3)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(acc_df)
print("**所有模型学到的任务表现总结：**")
print(acc_summary_df)
