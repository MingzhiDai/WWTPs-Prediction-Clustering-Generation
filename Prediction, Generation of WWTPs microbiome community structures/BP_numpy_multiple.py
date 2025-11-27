import numpy as np


# ReLU 激活函数
def relu(x):
    return np.maximum(0, x)

# ReLU 导数
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 激活函数（Sigmoid）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def min_max_normalize(X, min_val=0, max_val=1):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min) * (max_val - min_val) + min_val, X_min, X_max

def min_max_denormalize(X_normalized, X_min, X_max, min_val=0, max_val=1):
    return (X_normalized - min_val) / (max_val - min_val) * (X_max - X_min) + X_min

# 前向传播
def forward(X, weights, biases):
    activations = [X]
    for i in range(len(weights) - 1):  # 对于隐藏层
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        A = relu(Z)
        activations.append(A)

    # 输出层不使用激活函数（线性输出）
    Z = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(Z)
    return activations


# 初始化权重和偏置
# def initialize_weights(layers):
#     weights = []
#     biases = []
#     for i in range(len(layers) - 1):
#         W = np.random.rand(layers[i], layers[i + 1])
#         b = np.random.rand(layers[i + 1])
#         weights.append(W)
#         biases.append(b)
#     return weights, biases

# 初始化权重和偏置(Xavier 或 He 初始化)
def initialize_weights(layers):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        # Xavier 初始化 (适用于 Sigmoid, Tanh)
        W = np.random.randn(layers[i], layers[i+1]) * np.sqrt(1 / layers[i])
        b = np.zeros((1, layers[i+1]))
        weights.append(W)
        biases.append(b)
    return weights, biases

# 反向传播

# def backward(activations, y, weights, biases, learning_rate):
#     deltas = [activations[-1] - y]  # 输出层误差（线性输出层）
#
#     # 从后往前计算隐藏层的误差
#     for i in reversed(range(len(weights) - 1)):
#         delta = deltas[-1].dot(weights[i + 1].T) * relu_derivative(activations[i + 1])
#         deltas.append(delta)
#     deltas.reverse()
#
#     # 更新权重和偏置
#     for i in range(len(weights)):
#         weights[i] -= learning_rate * activations[i].T.dot(deltas[i])
#         biases[i] -= learning_rate * np.sum(deltas[i], axis=0)
#
#     return weights, biases


# 在反向传播时加入 L2 正则化
def backward(activations, y, weights, biases, learning_rate, lambda_reg=0.01):
    deltas = [activations[-1] - y]  # 输出层误差

    for i in reversed(range(len(weights) - 1)):
        delta = deltas[-1].dot(weights[i + 1].T) * relu_derivative(activations[i + 1])
        deltas.append(delta)
    deltas.reverse()

    for i in range(len(weights)):
        # 添加 L2 正则化项
        weights[i] -= learning_rate * (activations[i].T.dot(deltas[i]) + lambda_reg * weights[i])
        biases[i] -= learning_rate * np.sum(deltas[i], axis=0)

    return weights, biases


# 训练神经网络
def train(X, y, layers, epochs, learning_rate):
    weights, biases = initialize_weights(layers)

    for epoch in range(epochs):
        activations = forward(X, weights, biases)
        weights, biases = backward(activations, y, weights, biases, learning_rate)

        # 每100次迭代打印损失
        if epoch % 1 == 0:
            loss = np.mean((activations[-1] - y) ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights, biases


# 预测
def predict(X, weights, biases):
    activations = forward(X, weights, biases)
    return activations[-1]

# 示例数据
# X_train = np.array([[1], [2], [3], [4], [5]], dtype=float)
# y_train = np.array([[2], [4], [6], [8], [10]], dtype=float)
# 示例数据
X_train = np.array([[1], [2], [3], [4], [5]], dtype=float)
y_train = np.array([[2], [4], [6], [8], [10]], dtype=float)

# 归一化输入数据
X_train_normalized, X_min, X_max = min_max_normalize(X_train)

# 网络架构
layers = [1, 1]

# 训练模型
weights, biases = train(X_train_normalized, y_train, layers, epochs=100, learning_rate=0.01)
print("Weights:", weights)
print("Biases:", biases)

# 测试数据
X_test = np.array([[1]], dtype=float)

# 归一化测试数据
X_test_normalized = (X_test - X_min) / (X_max - X_min)

# 预测
y_pred_normalized = predict(X_test_normalized, weights, biases)

# 反归一化预测结果
y_pred = min_max_denormalize(y_pred_normalized, np.min(y_train), np.max(y_train))
print(f"预测结果: {y_pred}")