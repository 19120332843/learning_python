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

# cnn = Net()
# cnn = torch.load('./net/mobilenet1.pkl')
# print(cnn)
# for i in cnn.parameters():
#     print(i)

def hswish(x):
    out = x * F.relu6(x + 3, inplace = True) / 6
    return out

def my(x):
    out = x * F.relu(x + 4) / 8
    return out

m = nn.ReLU6()

x = np.linspace(-8, 8, 1000)
x = torch.from_numpy(x).type(torch.FloatTensor)
y1 = hswish(x)
y2 = m(x)
y3 = my(x)
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b')
plt.show()