import numpy as np
import math

class NeuralNetwork:
    def __init__(self, X, y, W, b, learning_rate, num_epochs):

        self.X = X  # 形状为 (4, 1)
        self.y = y  # 形状为 (4, 1)

        # 初始化神经网络参数（权重和偏置）
        np.random.seed(42)  # 固定随机数种子以获得可重复的结果
        self.W = np.random.randn(1, 1)  # 权重，形状为 (1, 1)
        self.b = np.random.randn(1)  # 偏置，形状为 (1,)

         # 定义超参数
        # self.learning_rate = 1e-10
        # self.num_epochs = 10

    # 前向传播：计算输出
    def forward(self, X, W, b):
        return np.dot(X, W) + b

    # 损失函数：均方误差
    def compute_loss(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    # 反向传播：计算梯度并更新参数
    def backward(self, X, y, y_hat, W, b, learning_rate):
        m = X.shape[0]
        dW = -(2 / m) * np.dot(X.T, (y - y_hat))
        db = -(2 / m) * np.sum(y - y_hat)
        W -= learning_rate * dW
        b -= learning_rate * db
        return W, b

    # 训练神经网络
    def train(self, X, y, W, b, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # 前向传播
            y_hat = self.forward(X, W, b)

            # 计算损失
            loss = self.compute_loss(y, y_hat)

            # 反向传播并更新参数
            W, b = self.backward(X, y, y_hat, W, b, learning_rate)

            # 每 100 个 epoch 打印一次损失
            if (epoch + 1) % 1 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss}')

        # 训练后的输出
        print("训练后的权重 W:", W)
        print("训练后的偏置 b:", b)

# 使用示例
# input_nodes = 784  # MNIST数据集的输入节点数
# hidden_nodes = 100  # 隐藏层节点数
# output_nodes = 10  # 输出层节点数(0-9的分类)
# learning_rate = 0.1
#
# # 初始化神经网络
# nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#
# # 训练神经网络，这里简单示例10次迭代
# for i in range(10):
#     # 假设train_X是输入数据，train_y是目标输出
#     nn.train(train_X, train_y)
#
# # 使用训练好的神经网络进行预测
# predictions = nn.query(test_X)