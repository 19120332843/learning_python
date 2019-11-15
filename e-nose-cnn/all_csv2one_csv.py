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

file_paths = []#F:\gitworkspace\python\e-nose-cnn\data
listdir("D:\\WeChat\\WeChat Files\\mzf14ihntts\\FileStorage\\File\\2019-11\\dataset\\train_set", file_paths)
# print(file_paths)

##csv add hearder
data = range(60)
data = pd.DataFrame(data)
data = data.values
data = list(map(list,zip(*data)))
data = pd.DataFrame(data)
data.to_csv(os.path.join("D:\\WeChat\\WeChat Files\\mzf14ihntts\\FileStorage\\File\\2019-11\\dataset", "train_set.csv"), encoding='utf_8_sig', index=False, header=0)

for i in file_paths:
    try:
        data = pd.read_csv(i, usecols=['0','1','2','3','4','5','6','7','8','9'])
        data = data.values
        data = list(map(list,zip(*data)))
        data = pd.DataFrame(data)
        data.to_csv(os.path.join("D:\\WeChat\\WeChat Files\\mzf14ihntts\\FileStorage\\File\\2019-11\\dataset", "train_set.csv"), encoding='utf_8_sig',index=False, header=0, mode = 'a+') 
    except Exception as e:
        print(i)