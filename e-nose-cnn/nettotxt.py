import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import sys
sys.path.append(r"F:\gitworkspace\DL-ZYNQ")
from cnndw import Net
# from cnndw import hswish
# import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = Net()
class hswish(nn.Module):
  def forward(self, x):
    out = x * F.relu(x + 3) / 4
    return out

cnn = torch.load('./net/mobilenetallrelu.pkl')
data = np.load('codedata/3times/new.npy')
label = np.load('codedata/3times/label.npy')
data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
label = torch.from_numpy(label).type(torch.int64)
data = data.view(700, 10, 1, 120)
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

list_file = []

for name, param in cnn.named_parameters():
	# print(name, param)
  print(len(param))
  for one_line in param:   
    list_file.append(one_line)  
  print(len(list_file))
  print(type(list_file))
  # print(list_file)
  arr_file = np.array(list_file)
  # print(arr_file)
  arr_file = arr_file.reshape(-1)
  print(arr_file)  


