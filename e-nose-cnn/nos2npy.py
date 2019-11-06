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
listdir("F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\woshu\\3", file_paths)
# print(file_paths)

data = range(120)
data = pd.DataFrame(data)
data = data.values
data = list(map(list,zip(*data)))
data = pd.DataFrame(data)
data.to_csv(os.path.join("F:\\gitworkspace\\python\\e-nose-cnn", "testnos2npy.csv"), encoding='utf_8_sig', index=False, header=0)

for i in file_paths:
    try:
        data = pd.read_csv(i, sep='\t', dtype=np.object,skiprows=52, header=None, engine='python')
        data = data.loc[0:119]
        data = data[range(10)]
        data = pd.DataFrame(data)
        data = data.values
        data = list(map(list,zip(*data)))
        data = pd.DataFrame(data)
        data.to_csv(os.path.join("F:\\gitworkspace\\python\\e-nose-cnn", "testnos2npy.csv"), encoding='utf_8_sig', index=False, header=0,mode = 'a+')#utf_8_sig
    except Exception as e:
        print(i)

df = pd.read_csv( "F:\\gitworkspace\\python\\e-nose-cnn\\testnos2npy.csv")
np.save('F:\\gitworkspace\\python\\e-nose-cnn\\testnos2npy.npy', df)
df = np.load('F:\\gitworkspace\\python\\e-nose-cnn\\testnos2npy.npy')
for i in range(6):
    print(df)