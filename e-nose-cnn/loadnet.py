import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import sys
sys.path.append(r"F:\gitworkspace\python\e-nose-cnn")
from cnndw import Net
from cnndw import hswish
import matplotlib.pyplot as plt

def Normlize(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Zmean = Z.mean(axis=1)
    #按列排序
    Zmax, Zmin = Zmax.reshape(-1, 1), Zmin.reshape(-1, 1)
    Zmean = Zmean.reshape(-1, 1)
    Z = (Z - Zmean) / (Zmax - Zmin)
    return Z

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#cuda:0
cnn = Net()
cnn = torch.load('./net/mobilenetallrelu.pkl')
data = np.load('new.npy')
label = np.load('codedata/3times/label.npy')
print(data)
print(data.shape)
data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
label = torch.from_numpy(label).type(torch.int64)
data = data.view(700, 10, 1, 120)
print(data)
label = label.squeeze()
label = label.to(device)
te_x = Variable(data)
te_y = Variable(label)
out1 = cnn(te_x)
predicted_test = torch.max(out1.data, 1)[1]
total = te_y.size(0)
print(total)
sum = 0 
for j in range(total):
  if predicted_test[j] == te_y[j]:
    sum += 1
print(sum)


# print(cnn)
# for i in cnn.parameters():
#     print(i)

# def hswish1(x):
#     out = x * F.relu6(x + 3, inplace = True) / 6
#     return out

# def my(x):
#     out = F.relu(x + 4) / 2
#     return out

# m = nn.ReLU6()

# x = np.linspace(-8, 8, 1000)
# x = torch.from_numpy(x).type(torch.FloatTensor)
# y1 = hswish1(x)
# y2 = m(x)
# y3 = my(x)
# plt.plot(x, y1, 'r')
# plt.plot(x, y2, 'g')
# plt.plot(x, y3, 'b')
# plt.show()