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
listdir("F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\train", file_paths)


# a 是计算file_paths的上限
# a = len(file_paths)
total = 0
mean = 0
d = 0
a = 75
print('=====================一共', a, '个样本。========================================')

# b 是计算下限
b = 0
for i in file_paths[0*75:1*75]:
    b += 1
    # b = a时跳出
    if (b == a):
        break
    c = b
    for j in file_paths[b:a]:
        one = b - 1
        two = c
        data1 = pd.read_csv(i)
        data1 = data1.values
        data1 = data1.reshape(-1)
        print(one, '=', data1)

        data2 = pd.read_csv(j)
        data2 = data2.values
        data2 = data2.reshape(-1)
        print(two, '=', data2)

        #data3是每个位置的差，data4是每个差的和
        data3 = data1
        data4 = 0
        for k in range(1200):
            data3[k] = data1[k] - data2[k]
            data4 += data3[k]**2           
        data4 = data4**(1/2)
        print(data4)
        c += 1
        mean += data4
        d += 1 
        if (data4 < 5):
            total += 1
        print('-----------------------------------------------------------------')
    print('=======================================================================')
print(total)
print(mean/d)
