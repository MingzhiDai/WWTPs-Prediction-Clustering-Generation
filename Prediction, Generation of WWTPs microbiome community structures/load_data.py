import pandas as pd
import numpy as np


def load_data(file_name):

    # 读取数据
    # data = pd.read_excel(file_name, index_col=0)  # index_col=0如果不设置，转换后csv文件第一列就会是索引0，1，2...
    # data.to_csv('Sample_information_final-Phylums.csv', encoding='utf-8')
    # df = pd.read_csv('Sample_information_final-Phylums.csv')
    Df = pd.read_csv(file_name)
    df = Df.values
    M = len(df)
    data_set = df[:, 1:]
    return Df, data_set, M

    # Feature = ['Annual average','Sampling month average','Annual','Sampling month']
    # Target = ['Proteobacteria','Bacteroidota','Patescibacteria','Unclassified','Firmicutes','Planctomycetota','Verrucomicrobiota',
    #           'Chloroflexi','Myxococcota','Others','Bdellovibrionota','Actinobacteriota','Acidobacteriota','Desulfobacterota',
    #           'Dependentiae','Cyanobacteria','Spirochaetota','Nanoarchaeota','Elusimicrobiota','SAR324 clade(Marine group B)','Armatimonadota']

    #
    # Feature = ['Annual average(℃)', 'Annual mean of daily maximum(℃)', 'Annual mean of daily minimum(℃)',
    #  'Sampling month average(℃)',
    #  'Sampling moment(℃)', 'Annual(mm)', 'Sampling month(mm)',
    #  'GDP per capita (dollars)', 'City population', 'Actual Inf rate (m3/d)',
    #  'HRT (hr) Plant', 'HRT (hr) Aeration tank', 'SRT (d)', 'BOD:(mg/l):Inf',
    #  'BOD:inf/(1+recycle ratio)', 'BOD:Aeration tank inf', 'BOD:Eff',
    #  'COD:(mg/l):Inf', 'COD:Inf/(1+recycle ratio)', 'COD:Aeration tank inf',
    #  'COD:Eff', 'F/M (kg BOD/kg MLSS-d)', 'NH4-N:(mg/l):Inf',
    #  'NH4-N:Aeration tank inf', 'NH4-N:Eff', 'TN:(mg/l):Inf',
    #  'TN:Aeration tank inf', 'TN:Eff', 'TP:(mg/l):Inf',
    #  'TP:Aeration tank inf', 'TP:Eff', 'Percentage', 'MLSS (mg/l)',
    #  'DO (mg/l)', 'pH', 'Mixed liquid temperature(℃)',
    #  'Conductivity (μS/cm)', 'SVI (ml/g)']
    #
    # Target = ['Proteobacteria', 'Bacteroidota',
    #    'Patescibacteria', 'Unclassified', 'Firmicutes', 'Planctomycetota',
    #    'Verrucomicrobiota', 'Chloroflexi', 'Myxococcota', 'Others',
    #    'Bdellovibrionota', 'Actinobacteriota', 'Acidobacteriota',
    #    'Desulfobacterota', 'Dependentiae', 'Cyanobacteria', 'Spirochaetota',
    #    'Nanoarchaeota', 'Elusimicrobiota', 'SAR324 clade(Marine group B)',
    #    'Armatimonadota']

    # feature_1 = df['Annual average']
    # feature_2 = df['Sampling month average']
    # feature_3 = df['Annual']
    # feature_4 = df['Sampling month']

    # feature_1 = df['Sampling month average']
    # feature_2 = df['Sampling month']
    # feature_3 = df['SRTd']
    # feature_4 = df['COD']
    # feature_5 = df['NH4N']
    # feature_6 = df['TP']
    #
    # target_1 = df['Proteobacteria']
    # target_2 = df['Bacteroidota']
    # target_3 = df['Patescibacteria']
    # target_4 = df['Unclassified']
    # target_5 = df['Firmicutes']
    # target_6 = df['Planctomycetota']
    # target_7 = df['Verrucomicrobiota']
    # target_8 = df['Chloroflexi']
    # target_9 = df['Myxococcota']
    # target_10 = df['Others']
    # target_11 = df['Bdellovibrionota']
    # target_12 = df['Actinobacteriota']
    # target_13 = df['Acidobacteriota']
    # target_14 = df['Desulfobacterota']
    # target_15 = df['Dependentiae']
    # target_16 = df['Cyanobacteria']
    # target_17 = df['Spirochaetota']
    # target_18 = df['Nanoarchaeota']
    # target_19 = df['Elusimicrobiota']
    # target_20 = df['SAR324 clade(Marine group B)']
    # target_21 = df['Armatimonadota']

    # 数据预处理，每n个取一个平均值
    # Feature_1 = []
    # Feature_2= []
    # Feature_3 = []
    # Feature_4 = []
    # Target_1 = []
    # Target_2 = []
    # Target_3 = []
    # Target_4 = []
    # Target_5 = []
    # Target_6 = []
    # Target_7 = []
    # Target_8 = []
    # Target_9 = []
    # Target_10 = []
    # Target_11 = []
    # Target_12 = []
    # Target_13 = []
    # Target_14 = []
    # Target_15 = []
    # Target_16 = []
    # Target_17 = []
    # Target_18 = []
    # Target_19 = []
    # Target_20 = []
    # Target_21 = []

    # n = 6  # 5
    # for i in range(0, len(feature_1), n):
    #     try:
    #         # 特征数据
    #         Feature_1_temp = round(sum([feature_1[i + j] for j in range(n)]) / n, 4) #3
    #         Feature_1.append(Feature_1_temp)
    #         # 特征数据
    #         Feature_2_temp = round(sum([feature_2[i + j] for j in range(n)]) / n, 4) #3
    #         Feature_2.append(Feature_2_temp)
    #
    #         Feature_3_temp = round(sum([feature_3[i + j] for j in range(n)]) / n, 4) #3
    #         Feature_3.append(Feature_3_temp)
    #         # 功率数据
    #         Feature_4_temp = round(sum([feature_4[i + j] for j in range(n)]) / n, 4) #3
    #         Feature_4.append(Feature_4_temp)
    #
    #         Target_1_temp = round(sum([target_1[i + j] for j in range(n)]) / n, 21)
    #         Target_1.append(Target_1_temp)
    #
    #         Target_2_temp = round(sum([target_2[i + j] for j in range(n)]) / n, 21)
    #         Target_2.append(Target_2_temp)
    #
    #         Target_3_temp = round(sum([target_3[i + j] for j in range(n)]) / n, 21)
    #         Target_3.append(Target_3_temp)
    #
    #         Target_4_temp = round(sum([target_4[i + j] for j in range(n)]) / n, 21)
    #         Target_4.append(Target_4_temp)
    #
    #         Target_5_temp = round(sum([target_5[i + j] for j in range(n)]) / n, 21)
    #         Target_5.append(Target_5_temp)
    #
    #         Target_6_temp = round(sum([target_6[i + j] for j in range(n)]) / n, 21)
    #         Target_6.append(Target_6_temp)
    #
    #         Target_7_temp = round(sum([target_7[i + j] for j in range(n)]) / n, 21)
    #         Target_7.append(Target_7_temp)
    #
    #         Target_8_temp = round(sum([target_8[i + j] for j in range(n)]) / n, 21)
    #         Target_8.append(Target_8_temp)
    #
    #         Target_9_temp = round(sum([target_9[i + j] for j in range(n)]) / n, 21)
    #         Target_9.append(Target_9_temp)
    #
    #         Target_10_temp = round(sum([target_10[i + j] for j in range(n)]) / n, 21)
    #         Target_10.append(Target_10_temp)
    #
    #         Target_11_temp = round(sum([target_11[i + j] for j in range(n)]) / n, 21)
    #         Target_11.append(Target_11_temp)
    #
    #         Target_12_temp = round(sum([target_12[i + j] for j in range(n)]) / n, 21)
    #         Target_12.append(Target_12_temp)
    #
    #         Target_13_temp = round(sum([target_13[i + j] for j in range(n)]) / n, 21)
    #         Target_13.append(Target_13_temp)
    #
    #         Target_14_temp = round(sum([target_14[i + j] for j in range(n)]) / n, 21)
    #         Target_14.append(Target_14_temp)
    #
    #         Target_15_temp = round(sum([target_15[i + j] for j in range(n)]) / n, 21)
    #         Target_15.append(Target_15_temp)
    #
    #         Target_16_temp = round(sum([target_16[i + j] for j in range(n)]) / n, 21)
    #         Target_16.append(Target_16_temp)
    #
    #         Target_17_temp = round(sum([target_17[i + j] for j in range(n)]) / n, 21)
    #         Target_17.append(Target_17_temp)
    #
    #         Target_18_temp = round(sum([target_18[i + j] for j in range(n)]) / n, 21)
    #         Target_18.append(Target_18_temp)
    #
    #         Target_19_temp = round(sum([target_19[i + j] for j in range(n)]) / n, 21)
    #         Target_19.append(Target_19_temp)
    #
    #         Target_20_temp = round(sum([target_20[i + j] for j in range(n)]) / n, 21)
    #         Target_20.append(Target_20_temp)
    #
    #         Target_21_temp = round(sum([target_21[i + j] for j in range(n)]) / n, 21)
    #         Target_21.append(Target_21_temp)
    #     except:
    #         pass

    # 分割数据集
    # M = len(feature_1)
    # data_set = np.array([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, target_1, target_2, target_3,target_4, target_5, target_6, target_7, target_8, target_9, target_10, target_11, target_12, target_13, target_14, target_15, target_16, target_17, target_18, target_19, target_20, target_21]).T
    #
    # return data_set, M