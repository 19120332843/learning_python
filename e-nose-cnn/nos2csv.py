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
listdir("F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\zhike\\1", file_paths)#F:\gitworkspace\python\e-nose-cnn\nos-data\3times7class\binglang\1
print(file_paths)
a = 0
for i in file_paths:
    try:
        #nos2csv
        data = pd.read_csv(i, sep='\t', dtype=np.object,
            skiprows=52, header=None, engine='python')
        # print(data)
        data = data.loc[0:119]
        # print(data)
        data = data[range(10)]
        # print(data)
        data.to_csv(os.path.join("F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\zhike\\2", "{0[6]}-{1}.csv".format(i.split("\\"), a)), encoding='utf_8_sig', index=False, header=1)#utf_8_sig
        a += 1
    except Exception as e:
        print(i)