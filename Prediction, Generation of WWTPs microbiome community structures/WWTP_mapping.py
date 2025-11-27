import pandas
import pandas as pd
import numpy as np
import csv
import xlrd

def get_coordinates(Data, target_array):
    """
    根据要查找的目标，返回其在excel中的位置
    data: excel数据,
    target: 要查找的目标
    return: 返回坐标列表
    """
    # Data_list = np.array(Data).tolist()
    for i in range(len(Data)):
        if (Data[i][1:] == target_array).all() == True:
            return i+1
    return []


# 读取excel文件
workbook = xlrd.open_workbook('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//New_data_2//clustering//WWTP_class--environmental and geographical features//The WWTP ranking information of each centroid.xlsx')
sheets_to_load = workbook.sheet_names()
Original_Data = pd.read_excel('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//New_data_2//clustering//Data_Phylums.xlsx', header=None).values
Data = np.array(Original_Data[1:][:])
# A = Data[0][1:]
data_new =[]
data_list = []
df_list = []
for i in range(len(sheets_to_load)):
    data = pd.read_excel('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//New_data_2//clustering//WWTP_class--environmental and geographical features//The WWTP ranking information of each centroid.xlsx', sheet_name = sheets_to_load[i], header=None).values
    # Label = np.zeros((len(data), 1), dtype=int)  #
    Label = []
    for j in range(len(data)):    #
        target_array = data[j]    #
        # target_array = target_array.split(",")
        coordinates = get_coordinates(Data, target_array)
        # print(f"{target_array}在第{coordinates-1}行")
        Label = np.append(Label, Data[coordinates-1][0])
    # data_new = np.append(Label, data, axis=1)  #
    # data_new = np.vstack((Label, data))
    # data_new = np.dstack((Label, data))
    Label_data = np.reshape(Label, (len(data), 1))
    for k in range(data.shape[1]):
        data_one = np.reshape(data[:, k], (data.shape[0], 1))
        Label_data = np.append(Label_data, data_one, axis=1)
    data_list.append(Label_data)
    df = pd.DataFrame(data_list[i], index=None, columns=['a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e',
                                                         'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e',
                                                         'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e',
                                                         'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e',
                                                         'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e',
                                                         'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e'])
    df_list.append(df)

with pd.ExcelWriter('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//New_data_2//clustering//WWTP_class--environmental and geographical features//The WWTP ranking information of each centroid-标签化.xlsx') as writer:
    for i in range(len(df_list)):
        df_list[i].to_excel(writer, sheet_name=sheets_to_load[i])



# data_1 = pd.read_excel('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//The WWTP ranking information of each centroid.xlsx', sheet_name = 'Sheet1', header=None).values
# data_2 = pd.read_excel('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//The WWTP ranking information of each centroid.xlsx', sheet_name = 'Sheet2', header=None).values
# data_3 = pd.read_excel('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//The WWTP ranking information of each centroid.xlsx', sheet_name = 'Sheet3', header=None).values

# writer.save()
# writer.close()

# with open('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//The WWTP ranking information of each centroid-标签化.csv', 'w',encoding='utf-8', newline='') as file:
#     writer =csv.writer(file)
#     # writer.writerow(header)
#     for i in range(len(data_3)):
#         writer.writerow(data_new[i])

# data_1[coordinates] = np.concatenate([[15675], data_1[coordinates]])


# Data = data_1
# for i in range(len(sheets_to_load)-1):
#     data = pd.read_excel('F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//The WWTP ranking information of each centroid.xlsx', sheet_name = sheets_to_load[i+1], header=None)
#     Data = np.vstack((Data, data))

# 打印坐标
# print(f"{target_array}在第{coordinates[0]}行")  # 张思德在第2行,第3列