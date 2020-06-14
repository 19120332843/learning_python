import os
import matplotlib.pyplot as plt
import numpy as np


datasize = 7
yangben = 40+66+18+70+290*7

# file = open(".\\codedata\\3times\\dataforzynq.txt", "r")
# fileint = open(".\\codedata\\3times\\dataforzynqint8.txt", "a+")
file = open(".\\codedata\\3times\\param.txt", "r")
fileint = open(".\\codedata\\3times\\paramint8.txt", "a+")
data = []
datafile = []
for i in file.readlines():
  datafile.append(float(i.strip('\n')))
datafile = np.array(datafile)


plt.clf()
#计算最大值
min_value = 0
max_value = 0
data = datafile[yangben: yangben+datasize]
max_value = max(data)
min_value = min(data)
# print(max_value)
# print(min_value)
max_abs = max(abs(min_value), max_value)
#计算原始分布Po我觉得这个是data

X=np.linspace(-3,3,datasize,endpoint=True)

Po = ((data/max_abs)*2048).astype(np.int)
plt.plot(X, Po)
# print(Po)
#计算P
Q = Po
num_bins = 2048
scale = 127/num_bins
#截断P

#将P量化到Q
Q = (Po*scale).astype(np.int)
# print(len(Q))
plt.plot(X, Q)

for i in range(datasize):
  s = str(Q[i])
  fileint.write(s + '\n')

plt.savefig('.\\codedata\\quanfig\\param\\test{}.jpg'.format(11))

file.close()
fileint.close()
