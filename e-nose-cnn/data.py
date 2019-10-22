import os
import pandas as pd
import numpy as np


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


file_paths = []
listdir("F:\\gitworkspace\\python\\e-nose-cnn\\binglang", file_paths)
print(file_paths)

for i in file_paths:
    # excel_data = pd.read_csv(i,sep=',',dtype=np.object,userows=[44, 104])
    # excel_data = pd.read_csv(i,sep=' ',dtype=np.object,skiprows=52,nrows=200,header=None)
    try:
        nos_data = pd.read_csv(i, sep='\t', dtype=np.object,
            skiprows=52, header=None, engine='python')
        #print(nos_data)
        nos_data2 = nos_data.loc[0:59]
        #print(nos_data2)
        nos_data3 = nos_data2[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        #print(nos_data3)
        nos_data3.to_csv(os.path.join("F:\\gitworkspace\\python\\e-nose-cnn\\data_csv", "{0[4]}-{0[5]}.csv".format(
            i.split("\\"))), encoding='utf_8_sig', index=False, header=0)  # 单纯utf8编码用excel打开会乱码
    except Exception as e:
        print(i)
