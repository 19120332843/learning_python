# test

## 说明

1. 每一个二级目录是一次参数改变；

2. 在此首先附上最原始的代码，每次参数改变需要记录改变什么参数；

3. 每次改变做10次测试；

```python
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import os
import sklearn
from sklearn.model_selection import train_test_split


def Normlize(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Zmean = Z.mean(axis=1)
    #按列排序
    Zmax, Zmin = Zmax.reshape(-1, 1), Zmin.reshape(-1, 1)
    Zmean = Zmean.reshape(-1, 1)
    Z = (Z - Zmean) / (Zmax - Zmin)
    return Z

def Data_Reading(Normalization=True):
    data = np.load('codedata/3times/dataset.npy')
    label = np.load('codedata/3times/label.npy')

    # Normalization
    data = Normlize(data)

    # myself or auto
    if Normalization:
        data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
        label = torch.from_numpy(label).type(torch.int64)

    # reshape
    data = data.view(700, 10, 1, 120)
    data = data.cpu().numpy()
    label = label.numpy()
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25)
    train_x = torch.from_numpy(train_x).type(torch.cuda.FloatTensor)
    test_x = torch.from_numpy(test_x).type(torch.cuda.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.int64)
    test_y = torch.from_numpy(test_y).type(torch.int64)
    return train_x, test_x, train_y, test_y

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3) / 6
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1d = nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = (1, 3), stride = 1, groups = 10)#(120 - 3)/1 + 1 = 118
        self.conv1p = nn.Conv2d(in_channels = 10, out_channels = 6, kernel_size = 1, stride = 1, groups = 1)
        self.hswish1 = hswish()
        self.conv2d = nn.Conv2d(in_channels = 6, out_channels = 6, kernel_size = (1, 4), stride = 1, groups = 6)#(59 - 4)/1 + 1 = 56
        self.conv2p = nn.Conv2d(in_channels = 6, out_channels = 10, kernel_size = 1, stride = 1, groups = 1)
        self.hswish2 = hswish()
        self.fc1 = nn.Linear(10*28, 7)#427
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.conv1d(x)
        x = F.relu6(x)
        x = self.conv1p(x)
        x = self.hswish1(x)
        x = F.max_pool2d(x, (1, 2))
        x = self.conv2d(x)
        x = F.relu6(x)
        x = self.conv2p(x)
        x = self.hswish2(x)
        x = F.max_pool2d(x, (1, 2))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x#


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#cuda:0

    print(device)

    cnn = Net()
    print(cnn)
    cnn.to(device)

    #sgd -> stochastic gradient descent
    lrr = 0.01
    mom = 0.9
    optimizer = optim.SGD(cnn.parameters(), lr=lrr, momentum=mom)#
    loss_func = nn.CrossEntropyLoss()#CrossEntropyLoss()

    train_x, test_x, train_y, test_y = Data_Reading(Normalization=1)
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)

    #train
    sum = 0
    max = 0
    batch_size = 21
    tr_x = Variable(train_x)
    tr_y = Variable(train_y)
    for epoch in range(200):
        running_loss = 0.0
        for i in range(0,(int)(len(train_x)/batch_size)):
            t_x = Variable(train_x[i*batch_size:i*batch_size+batch_size])
            t_y = Variable(train_y[i*batch_size:i*batch_size+batch_size])
            out = cnn(t_x)
            loss = loss_func(out, t_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss = running_loss / 25

        out = cnn(tr_x)
        predicted_train = torch.max(out.data, 1)[1]
        total_train = tr_y.size(0)#总数
        for j in range(tr_y.size(0)):
            if predicted_train[j] == tr_y[j]:
                sum += 1

        print('total_train:{}, accuracy:{}, sum:{}'.format(total_train, sum / total_train, sum))
        sum = 0

        if (sum / total_train > 0.85) :
            optimizer = optim.SGD(cnn.parameters(), lr=lrr/10, momentum=mom/4)
        elif (sum / total_train > 0.95) :
            optimizer = optim.SGD(cnn.parameters(), lr=lrr/10/10, momentum=mom/4/4)#momentum=0


        print('Epoch[{}], loss: {:.8f}'.format(epoch + 1, running_loss))

    #test
        te_x = Variable(test_x)
        te_y = Variable(test_y)
        out1 = cnn(te_x)
        predicted_test = torch.max(out1.data, 1)[1]#.data.squeeze()
        total = te_y.size(0)

        for j in range(te_y.size(0)):
            if predicted_test[j] == te_y[j]:
                sum += 1

        if(max < sum/total):
            max = sum/total
            maxepoch = epoch + 1
            torch.save(cnn, './net/mobilenet.pkl')
        print('total:{}, accuracy:{}, sum:{}, max={}, maxepoch={}'.format(total, sum / total, sum, max, maxepoch))
        print('=============================================================================')
        sum = 0
```

## 1

```python
class myrelu(nn.Module):
    def forward(self, x):
        out = x * F.relu(x + 5) / 16
        return x
```

```python
momdel = 8
```

|序号|结果|网络名字|
|:-:|:-:|:-:|
|0|max=0.8857142857142857, maxepoch=194||
|1|max=0.9314285714285714, maxepoch=132||
|2|max=0.9428571428571428, maxepoch=172|cnndw_myrelu.pkl|
|3|max=0.8914285714285715, maxepoch=137||
|4|max=0.92, maxepoch=186||
|5|max=0.8571428571428571, maxepoch=117||
|6|max=0.8742857142857143, maxepoch=163||
|7|max=0.9028571428571428, maxepoch=85||
|8|max=0.8628571428571429, maxepoch=185||
|9|max=0.9428571428571428, maxepoch=158||

## 2

```python
out = x * F.relu(x + 4) / 8
```

```python
x = relu(x) #第一和三层卷积使用
```

|序号|结果|网络名字|
|:-:|:-:|:-:|
|0|max=0.9314285714285714, maxepoch=170|cnndw_allrelu.pkl|
|1|max=0.84, maxepoch=122||
|2|max=0.9028571428571428, maxepoch=127||
|3|max=0.8685714285714285, maxepoch=178||
|4|max=0.8285714285714286, maxepoch=123||
|5|max=0.9542857142857143, maxepoch=189|cnndw_allrelu0.pkl|
|6|max=0.8742857142857143, maxepoch=133||
|7|max=0.9314285714285714, maxepoch=124||
|8|max=0.88, maxepoch=107||
|9|max=0.9257142857142857, maxepoch=130||
