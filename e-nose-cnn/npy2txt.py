import os
import pandas as pd
import csv
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
list_file = []
# 输入数据，700*10*120
data = np.load('codedata/3times/label.npy')

#逐行读取Excel文件中的每一行append列表中
for one_line in data:   
  list_file.append(one_line)  
print(len(list_file))
print(type(list_file))
arr_file = np.array(list_file)  #转换为矩阵形式
print(type(arr_file))
arr_file = arr_file.reshape(-1)
print(type(arr_file))
print(len(arr_file))
print(len(arr_file))

file = open(".\\codedata\\3times\\labelforzynq.txt", "w")
for i in range(int(len(arr_file))):
  # arr_file[i] = ':.5f'.format(arr_file[i])
  s = str(arr_file[i]).replace('[','').replace(']','')
  file.write(s + '\n')#file.write(str(arr_file[i*6:i*6+6]) + '\n')
file.close()