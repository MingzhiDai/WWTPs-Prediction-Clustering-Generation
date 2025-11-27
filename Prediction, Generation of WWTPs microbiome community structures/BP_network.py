# BP模块 借助PyTorch实现
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前使用的设备: {device}')

# 引入了遗传算法参数的BP模型
class GABP_net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, GA_parameter):
        super(GABP_net, self).__init__()
        # 构造隐含层和输出层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)
        # 给定网络训练的初始权值和偏执等
        self.hidden.weight = torch.nn.Parameter(GA_parameter[0])
        self.hidden.bias = torch.nn.Parameter(GA_parameter[1])
        self.output.weight = torch.nn.Parameter(GA_parameter[2])
        self.output.bias = torch.nn.Parameter(GA_parameter[3])

    def forward(self, x):
        # 前向计算
        hid = torch.sigmoid(self.hidden(x))
        out = torch.sigmoid(self.output(hid))
        return out

# 传统的BP模型
class ini_BP_net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(ini_BP_net, self).__init__()
        # 构造隐含层和输出层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 前向计算
        hid = torch.sigmoid(self.hidden(x))
        out = torch.sigmoid(self.output(hid))
        return out

def train(model, epochs, learning_rate, x_train, y_train):
    """
    :param model: 模型
    :param epochs: 最大迭代次数
    :param learning_rate: 学习率
    :param x_train: 训练数据（输入）
    :param y_train: 训练数据（输出）
    :return: 最终的 loss 值（MSE）
    """
    # 定义损失函数和优化器
    loss_fc = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    loss_list = []

    for i in range(epochs):
        model.train()
        # 前向计算
        data = model(x_train)
        # 计算误差
        loss = loss_fc(data, y_train)
        loss_list.append(loss.item())  # 使用 .item() 获取标量值
        # 更新梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

    return loss_list

# BP网络调试模块
if __name__ == "__main__":
    n_feature = 2
    n_hidden = 5
    n_output = 1800

    # 构建模型并移动到 GPU
    model = ini_BP_net(n_feature, n_hidden, n_output).to(device)

    # 测试数据准备
    x_train = torch.rand(5, n_feature).to(device)  # 将输入数据移动到 GPU
    y_label = torch.rand(5, n_output).to(device)   # 将标签数据移动到 GPU

    # 训练
    learn_rate = 1e-2
    loss_ls = train(model, 1000, learn_rate, x_train, y_label)

    # 可视化损失曲线
    plt.plot(loss_ls)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()