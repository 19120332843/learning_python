import os
import matplotlib.pyplot as plt
import numpy as np

#read data and param
paramfile = open(".\\codedata\\3times\\paramint8.txt", "r")
datafile = open(".\\codedata\\3times\\dataforzynqint8.txt", "r")
param = []
data = []
for i in paramfile.readlines():
  param.append(float(i.strip('\n')))
param = np.array(param)
print(param)
for i in datafile.readlines():
  data.append(float(i.strip('\n')))
data = np.array(data)
# print(data)
data = data.reshape(700, 10, 1, 120)

win1d = param[0 : 30]

win1d = win1d.reshape(10, 1, 3)
print(win1d)
win1b = param[30 : 40]
print(win1b)
# def conv1(din, win, bin)

