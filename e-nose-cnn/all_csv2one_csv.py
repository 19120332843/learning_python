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
listdir("F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\3\\train", file_paths)
# print(file_paths)

##csv add hearder
data = range(120)
data = pd.DataFrame(data)
data = data.values
data = list(map(list,zip(*data)))
data = pd.DataFrame(data)
data.to_csv(os.path.join("F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\3", "trainset2.csv"), encoding='utf_8_sig', index=False, header=0)

for i in file_paths:
    try:
        data = pd.read_csv(i, usecols=['0','1','2','3','4','5','6','7','8','9'])
        data = data.values
        data = list(map(list,zip(*data)))
        data = pd.DataFrame(data)
        data.to_csv(os.path.join("F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\3", "trainset2.csv"), encoding='utf_8_sig',index=False, header=0, mode = 'a+') 
    except Exception as e:
        print(i)