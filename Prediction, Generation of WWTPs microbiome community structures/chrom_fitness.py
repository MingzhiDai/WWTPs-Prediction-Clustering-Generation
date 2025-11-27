# 适应度计算模块
# 功能：传入一个编码，返回一个适应度值
from torchvision.transforms import transforms
import torch
import BP_network
import numpy as np

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前使用的设备: {device}')

# 计算MSE
def MSE(X, Y):
    X = np.array(X, dtype=np.complex128).flatten()
    Y = np.array(Y, dtype=np.complex128).flatten()
    if len(X) != len(Y):
        print("Wrong!")
    n = len(X)
    Wc = 0
    for i in range(n):
        Wc = Wc + (X[i] - Y[i]) * (X[i] - Y[i])
    return Wc / (2 * len(X))

def calculate_fitness(code, n_feature, n_hidden, n_output, epochs, learning_rate, x_train, y_train):
    """
    :param code: 染色体编码
    :param n_feature: 输入层个数
    :param n_hidden: 隐含层个数
    :param n_output: 输出层个数
    :param epochs: 最多迭代次数
    :param learning_rate: 学习率
    :param x_train: 训练（输入）数据
    :param y_train: 训练（输出）数据
    :return: fitness 适应度值
    """
    Parameter = code[:]
    # 参数提取
    hidden_weight = Parameter[0:n_feature * n_hidden]
    hidden_bias = Parameter[n_feature * n_hidden: n_feature * n_hidden + n_hidden]
    output_weight = Parameter[n_feature * n_hidden + n_hidden: n_feature * n_hidden + n_hidden + n_hidden * n_output]
    output_bias = Parameter[n_feature * n_hidden + n_hidden + n_hidden * n_output: n_feature * n_hidden + n_hidden + n_hidden * n_output + n_output]

    # 类型转换并移动到 GPU
    tensor_tran = transforms.ToTensor()
    hidden_weight = tensor_tran(np.array(hidden_weight).reshape((n_hidden, n_feature))).to(torch.float32).to(device)
    hidden_bias = tensor_tran(np.array(hidden_bias).reshape((1, n_hidden))).to(torch.float32).to(device)
    output_weight = tensor_tran(np.array(output_weight).reshape((n_output, n_hidden))).to(torch.float32).to(device)
    output_bias = tensor_tran(np.array(output_bias).reshape((1, n_output))).to(torch.float32).to(device)

    # 形状转换
    hidden_weight = hidden_weight.reshape((n_hidden, n_feature))
    hidden_bias = hidden_bias.reshape(n_hidden)
    output_weight = output_weight.reshape((n_output, n_hidden))
    output_bias = output_bias.reshape(n_output)

    # 带入模型计算
    GA = [hidden_weight, hidden_bias, output_weight, output_bias]
    BP_model = BP_network.GABP_net(n_feature, n_hidden, n_output, GA).to(device)  # 将模型移动到 GPU
    x_train = x_train.to(device)  # 将输入数据移动到 GPU
    y_train = y_train.to(device)  # 将标签数据移动到 GPU

    # 训练模型
    loss = BP_network.train(BP_model, epochs, learning_rate, x_train, y_train)

    # 计算适应度
    prediction = BP_model(x_train)
    fitness = 10 - MSE(prediction.detach().cpu().numpy(), y_train.detach().cpu().numpy())  # 将结果移回 CPU
    return np.round(fitness, 4)