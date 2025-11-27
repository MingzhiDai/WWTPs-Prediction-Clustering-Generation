# 个人编写的遗传算法依赖
import chrom_code
import chrom_mutate
import chrom_cross
import chrom_select
import chrom_fitness
import BP_network
from load_data import load_data
# 引入数据分析包
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
import torch
import time
from torchvision.transforms import transforms
from scipy.stats import pearsonr
# import lime.lime_tabular
import torch.nn.functional as nnf
import torch.nn as nn
import argparse
import csv
import os
from Generative_AI import Generative_AI

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600

# 检查是否有可用的 GPU
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前使用的设备 Current Device in Use: {device}')

parser = argparse.ArgumentParser()
parser.add_argument("--file")
parser.add_argument("--generated_file")
parser.add_argument("--n_feature", type=int)
parser.add_argument("--n_hidden", type=int)
parser.add_argument("--n_output", type=int)
parser.add_argument("--num_epoch", type=int)
parser.add_argument("--learn_rate", type=float)
parser.add_argument("--population_size", type=int)
# parser.add_argument("--chrom_len", type=int)
parser.add_argument("--p_cross", type=float)
parser.add_argument("--p_mutate", type=float)
parser.add_argument("--maxgen", type=int)
parser.add_argument("--output_dir")
parser.add_argument("--test_instance", type=int)
args = parser.parse_args()
#
# # 参数使用
file_name = args.file   # 原文件名称
file_name_generate = args.generated_file  # 生成式文件名称
n_feature = args.n_feature   # 输入水厂特征数目
n_hidden = args.n_hidden   # 神经网络隐层节点数目
n_output = args.n_output   # 输出菌群特征数目
num_epoch = args.num_epoch   # 神经网络迭代次数
learn_rate = args.learn_rate   # 神经网络学习率
population_size = args.population_size    # 种群规模
# chrom_len = args.chrom_len
p_cross = args.p_cross   # 差分进化算法交叉率
p_mutate = args.p_mutate  # 差分进化算法变异率
maxgen = args.maxgen
output_dir = args.output_dir  # 输出文件夹路径
test_instance = args.test_instance # 指定测试实例序号

start_time = time.time()

# 读取数据
# df = pd.read_excel('10.xlsx')
# file_name = 'Sample_information_final-Phylums.csv'
# file_name_generate = 'Generated_data_Phylums.csv'
# file_name = 'Sample_information_final-Class.csv'
# file_name_generate = 'Generated_data_Class.csv'
# file_name = 'Sample_information_final-Order.csv'
# file_name_generate = 'Generated_data_Order.csv'
# file_name = 'D://BaiduNetdiskDownload//Sample_information_final-Phylums.csv'
# file_name_generate = 'D://BaiduNetdiskDownload//Generated_data_Phylums.csv'

# file_name = 'Sample_information_final-Phylums.csv'
# file_name_generate = 'Generated_data_10.csv'
Df, data_set, M = load_data(file_name)
# Df_generate, data_set_generate, M_generate = load_data(file_name_generate)
Df_generate, data_set_generate, M_generate = Generative_AI(file_name)
Data_generate = data_set_generate[:100, :]
# Generated_data =
# env_T = df['环境温度']
# hum_T = df['环境湿度']
# wind_v = df['风速（机械）']
# eff = df['有功功率限制值']

# 分割数据集

train_set = data_set[:int(M * 0.8), :]
# cv_set = data_set[int(M * 0.6):int(M * 0.8), :]
test_set = data_set[int(M * 0.8):, :]

# print("训练集维度：", train_set.shape)
#
# print("测试集维度：", test_set.shape)

# 首先对数据进行归一化处理
scale_train = MinMaxScaler()
scale_test = MinMaxScaler()
train_set_norm = scale_train.fit_transform(train_set)
train_set_norm = np.vstack((train_set_norm, Data_generate))    # 融合Generate数据
# cv_set_norm = scale.fit_transform(cv_set)
test_set_norm = scale_test.fit_transform(test_set)


print("训练集维度 TrainSet Dimensions：", train_set_norm.shape)
print("测试集维度 TestSet Dimensions：", test_set_norm.shape)
print('数据准备完成 Data Preparation Complete...')

# 首先使用经典BP神经网络
# =========================== #
print("开始经典BP训练 Start Classic BP Training...")
# n_feature = 37
# n_hidden = 8
# n_output = 21
# num_epoch = 1500  # 1500
# learn_rate = 1e-2
BP_net = BP_network.ini_BP_net(n_feature, n_hidden, n_output)
x_train = train_set_norm[:, :n_feature]  # 711x4  [0,1,2,3]
y_train = train_set_norm[:, n_feature:].reshape(train_set_norm.shape[0], n_output)  # n_feature:,1   711x21
# print(x_train.shape,y_train.shape)

# 格式转换
tensor_tran = transforms.ToTensor()
x_train = tensor_tran(x_train).to(torch.float).reshape(x_train.shape[0], n_feature)  # 3
y_train = tensor_tran(y_train).to(torch.float).reshape(y_train.shape[0], n_output)  # 1
# 进行训练
BP_lossList = BP_network.train(BP_net, num_epoch, learn_rate, x_train, y_train)
# 测试集
x_test = tensor_tran(test_set_norm[:, :n_feature]).to(torch.float).reshape(test_set_norm[:, :n_feature].shape[0], n_feature)  # :3,0,:3,3
y_test = test_set_norm[:, n_feature:].reshape(test_set_norm.shape[0], n_output)  # 3,0,1
# 预测结果
BP_prediction = BP_net(x_test).detach().numpy()
# 拼接输入和预测结果
BP_pred_full = np.hstack([test_set_norm[:, :n_feature],BP_prediction])

# 使用测试集的 scaler 反归一化
BP_pred_recovered = scale_test.inverse_transform(BP_pred_full)

# 提取输出部分（真实值对应位置）
BP_prediction_original = BP_pred_recovered[:, n_feature:]

print("经典BP训练完成 Classic BP Training Completed...")
# ================================= #

# 基于遗传算法优化的BP神经网络 #
# ================================= #
print("开始进行遗传优化 Genetic Optimization Starting...")
chrom_len = n_feature * n_hidden + n_hidden + n_hidden * n_output + n_output  # 染色体长度
# size = 15  # 种群规模
bound = np.ones((chrom_len, 2))  # 2代表每个参数的上界和下界
sz = np.array([[-1, 0], [0, 1]])
bound = np.dot(bound, sz)  # 各基因取值范围
# p_cross = 0.4  # 交叉概率
# p_mutate = 0.01  # 变异概率
# maxgen = 20
# 遗传最大迭代次数

chrom_sum = []  # 种群，染色体集合
for i in range(population_size):
    chrom_sum.append(chrom_code.code(chrom_len, bound))
account = 1  # 遗传迭代次数计数器
best_fitness_ls = []  # 每代最优适应度
ave_fitness_ls = []  # 每代平均适应度
best_code = []  # 迭代完成适应度最高的编码值

# 适应度计算
fitness_ls = []
for i in range(population_size):
    fitness = chrom_fitness.calculate_fitness(chrom_sum[i], n_feature, n_hidden, n_output,
                                              num_epoch, learn_rate, x_train, y_train)
    fitness_ls.append(fitness)
# 收集每次迭代的最优适应值和平均适应值
fitness_array = np.array(fitness_ls).flatten()
fitness_array_sort = fitness_array.copy()
fitness_array_sort.sort()
best_fitness = fitness_array_sort[-1]
best_fitness_ls.append(best_fitness)
ave_fitness_ls.append(fitness_array.sum() / population_size)

while True:
    # 选择算子
    chrom_sum = chrom_select.select(chrom_sum, fitness_ls)
    # 交叉算子
    chrom_sum = chrom_cross.cross(chrom_sum, population_size, p_cross, chrom_len, bound)
    # 变异算子
    chrom_sum = chrom_mutate.mutate(chrom_sum, population_size, p_mutate, chrom_len, bound, maxgen, account + 1)
    # 适应度计算
    fitness_ls = []
    for i in range(population_size):
        fitness = chrom_fitness.calculate_fitness(chrom_sum[i], n_feature, n_hidden, n_output,
                                                  num_epoch, learn_rate, x_train, y_train)
        fitness_ls.append(fitness)
    print(fitness_ls)
    # 收集每次迭代的最优适应值和平均适应值
    fitness_array = np.array(fitness_ls).flatten()
    fitness_array_sort = fitness_array.copy()
    fitness_array_sort.sort()
    best_fitness = fitness_array_sort[-1]  # 获取最优适应度值
    best_fitness_ls.append(best_fitness)
    ave_fitness_ls.append(fitness_array.sum() / population_size)
    # 计数器加一
    print(f" 第{account}/{maxgen}次遗传迭代完成! {account}/{maxgen}th genetic iteration completed!")
    account = account + 1
    if account == (maxgen + 1):
        index = fitness_ls.index(max(fitness_ls))  # 返回最小值的索引
        best_code = chrom_sum[index]  # 通过索引获得对于染色体
        # print(fitness_ls)
        print(best_fitness_ls)
        break

# 参数提取
hidden_weight = best_code[0:n_feature * n_hidden]
hidden_bias = best_code[n_feature * n_hidden:
                        n_feature * n_hidden + n_hidden]
output_weight = best_code[n_feature * n_hidden + n_hidden:
                          n_feature * n_hidden + n_hidden + n_hidden * n_output]
output_bias = best_code[n_feature * n_hidden + n_hidden + n_hidden * n_output:
                        n_feature * n_hidden + n_hidden + n_hidden * n_output + n_output]
# 类型转换
tensor_tran = transforms.ToTensor()
hidden_weight = tensor_tran(np.array(hidden_weight).reshape((n_hidden, n_feature))).to(torch.float32)
hidden_bias = tensor_tran(np.array(hidden_bias).reshape((1, n_hidden))).to(torch.float32)
output_weight = tensor_tran(np.array(output_weight).reshape((n_output, n_hidden))).to(torch.float32)
output_bias = tensor_tran(np.array(output_bias).reshape((1, n_output))).to(torch.float32)
# 形状转换
hidden_weight = hidden_weight.reshape((n_hidden, n_feature))
hidden_bias = hidden_bias.reshape(n_hidden)
output_weight = output_weight.reshape((n_output, n_hidden))
output_bias = output_bias.reshape(n_output)
GA = [hidden_weight, hidden_bias, output_weight, output_bias]

gaBP_net = BP_network.GABP_net(n_feature, n_hidden, n_output, GA)
gaBP_lossList = BP_network.train(gaBP_net, num_epoch, learn_rate, x_train, y_train)
gaBP_prediction = gaBP_net(x_test).detach().numpy()
# 拼接输入和预测结果
gaBP_pred_full = np.hstack([test_set_norm[:, :n_feature], gaBP_prediction])

# 使用测试集的 scaler 反归一化
gaBP_pred_recovered = scale_test.inverse_transform(gaBP_pred_full)

# 提取输出部分（真实值对应位置）
gaBP_prediction_original = gaBP_pred_recovered[:, n_feature:]

end_time = time.time()
# test_instance = 2
output_path = os.path.join(output_dir, "Result_Generative.txt")
log_path = os.path.join(output_dir, "Gabp_log.xlsx")
csv_path = os.path.join(output_dir, "Test_instance_output.csv")

# 测试实例写入
if test_instance > len(x_test):
    print("Test instance index out of range!")
else:
    with open(csv_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(Df.columns[n_feature+1:])  # 写入表头
        Test_instance_output = gaBP_prediction_original[test_instance,:]
        writer.writerows([Test_instance_output])

print("遗传优化完成 Genetic optimization completed...")
#
print(f"程序用时 Program duration：{end_time - start_time} s")
# =================================== #

# 对两种算法的误差评价
loss_fc = torch.nn.MSELoss(reduction="sum")
y_test_ = tensor_tran(y_test).to(torch.float).reshape(y_test.shape[0], n_output)
# np.set_printoptions(threshold=sys.maxsize)  # 全部输出
BP_error = loss_fc(BP_net(x_test), y_test_).detach().numpy()/n_output
BP_TEST = BP_net(x_test)
# print("BP算法预测值为：", BP_net(x_test), "BP算法观测值为：", y_test)
gaBP_error = loss_fc(gaBP_net(x_test), y_test_).detach().numpy()/n_output
print("BP算法误差为 BP algorithm error：", BP_error, "\nGABP算法误差为 GABP algorithm error：", gaBP_error)

# import subprocess
# import os
# if BP_error <= gaBP_error:
#     print("模型误差不满足要求，重新运行脚本 Model error does not meet requirements; rerun script……")
#
#     # 使用相同的 Python 执行当前脚本
#     subprocess.call([sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
#
#     # 退出当前进程
#     sys.exit(0)
#
# print("误差满足要求，继续执行后续程序 Error meets requirements; proceed with subsequent steps")

BP_net = BP_network.ini_BP_net(n_feature, n_hidden, n_output)
feature_weight_tensor = BP_net.hidden.weight.abs().mean(dim=0)
feature_weight = feature_weight_tensor.detach().numpy()
print(feature_weight)

feature_weight_tensor2 = BP_net.output.weight.abs().mean(dim=1)
feature_weight2 = feature_weight_tensor2.detach().numpy()
print(feature_weight2)

R = []
for i in range(len(y_test)):
    R.append(pearsonr(y_test[i], gaBP_prediction[i])[0])
R_array = np.array(R)

# explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values,feature_names = Data.columns,class_names =['target'] ,discretize_continuous=True)
# 将算法结果写入log.txt #
# f = open('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//DE-BP python//gabp(新)//成果图//New_resluts_2//Phylum_generative=20//log.txt', 'a', encoding='UTF-8')
f = open(output_path, 'a', encoding='UTF-8')
f.write("神经网络拓扑结构为 Neural network topology：" + str(n_feature) + ' ' + str(n_hidden) + ' ' + str(n_output) + '\n')
f.write("网络迭代次数 Network iteration count：" + str(num_epoch) + '\n')
f.write("遗传迭代所获得的最优权值为 Optimal weights obtained via genetic iteration：" + ", ".join([str(float(num)) for num in best_code]) + "\n")
f.write("DE-BP算法预测值为 DE-BP algorithm prediction values\n" + str(gaBP_prediction.flatten()) + '\n')
f.write("程序用时 Running time：" + str(end_time - start_time) + '\n')
f.write(f"BP算法误差 BP algorithm error：{BP_error} \nGABP算法误差 GABP algorithm error：{gaBP_error}\n\n")
# f.write("Observation_values：\n" + str(y_test) + '\n\n')
# f.write("Prediction_values：\n" + str(gaBP_prediction) + '\n\n')
# f.write("WWTP_feature_weight：\n" + str(feature_weight) + '\n')
f.close()

df_1 = pd.DataFrame(test_set[:,n_feature:], index=None)
df_2 = pd.DataFrame(gaBP_prediction_original, index=None)
df_3 = pd.DataFrame(R_array, index=None)
df_4 = pd.DataFrame(feature_weight, index=None)

# with pd.ExcelWriter('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//DE-BP python//gabp(新)//成果图//New_resluts_2//Phylum_generative=20//Gabp+GAN_log.xlsx') as writer:
with pd.ExcelWriter(log_path) as writer:
    df_1.to_excel(writer, sheet_name='Observation_values')
    df_2.to_excel(writer, sheet_name='Prediction_values')
    df_3.to_excel(writer, sheet_name='Pearson correlation coefficient')
    df_4.to_excel(writer, sheet_name='WWTP_feature_weight')

# 可视化 #
for i in range(n_output):
    plt.figure()
    plt.plot(BP_prediction_original[:, i], label='BP prediction', c='r')
    plt.plot(test_set[:,n_feature:][:, i], label='Real data', c='b')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "BP_prediction_") + str(i) +'.png', dpi=600)

for i in range(n_output):
    plt.figure()
    plt.plot(gaBP_prediction_original[:, 0], label='DE-BP prediction', c='r')
    plt.plot(test_set[:,n_feature:][:, 0], label='Real data', c='b')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "DE-BP_prediction_") + str(i) +'.png', dpi=600)

plt.figure()
plt.plot(BP_lossList, c='b')
plt.ylabel("BP error drop curve")
plt.savefig(os.path.join(output_dir, "BP_error_drop_curve.png"), dpi=600)

plt.figure()
plt.plot(gaBP_lossList, c='b')
plt.ylabel("DE-BP error drop curve")
plt.savefig(os.path.join(output_dir, "DE-BP_error_drop_curve.png"), dpi=600)

plt.show()

