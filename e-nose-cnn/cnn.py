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

def Normlize2(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Zmean = Z.mean(axis=1)
    #按列排序
    Zmax, Zmin = Zmax.reshape(-1, 1), Zmin.reshape(-1, 1)
    Zmean = Zmean.reshape(-1, 1)
    Z = (Z - Zmin) / (Zmax - Zmin)
    return Z


def Data_Reading(Normalization=True):
    data = np.load('e-nose-cnn/codedata/3times/dataset.npy')
    label = np.load('e-nose-cnn/codedata/3times/label.npy')

    # Normalization
    data = Normlize(data)

    # myself or auto
    if Normalization:
        data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
        label = torch.from_numpy(label).type(torch.int64)

    # reshape
    data = data.view(700, 1, 10, 120)
    
    data = data.cpu().numpy()
    label = label.numpy()    
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25) 
    train_x = torch.from_numpy(train_x).type(torch.cuda.FloatTensor)
    test_x = torch.from_numpy(test_x).type(torch.cuda.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.int64)
    test_y = torch.from_numpy(test_y).type(torch.int64)

    # permutation = np.random.permutation(train_y.shape[0])
    # train_x = train_x[permutation, :, :, :]
    # print('---------------------------------------------------------------------------')
    # train_y = train_y[permutation]
    return train_x, test_x, train_y, test_y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,(3,3),stride=(1,1))#(10 - 3)/1+1=8 :（120 - 3）/1 + 1 = 118 
        self.conv2 = nn.Conv2d(6,10,(4,4),stride=(2,1))#(8 - 4)/2+1 = 3 : (118-4)/1+1 = 115
        self.fc1 = nn.Linear(10*3*115,7)#427
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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
    lrr = 0.008
    optimizer = optim.SGD(cnn.parameters(), lr=lrr, momentum=0.8)#
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
    for epoch in range(160):
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

        if (sum / total_train > 0.92) :
            optimizer = optim.SGD(cnn.parameters(), lr=lrr/10, momentum=0.8/4)
        elif (sum / total_train > 0.98) :
            optimizer = optim.SGD(cnn.parameters(), lr=lrr/10/10, momentum=0)#momentum=0


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
        print('total:{}, accuracy:{}, sum:{}, max={}, maxepoch={}'.format(total, sum / total, sum, max, maxepoch))
        print('=============================================================================')
        sum = 0