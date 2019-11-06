import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import os 


def Normlize(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Zmean = Z.mean(axis=1)
    #按列排序
    Zmax, Zmin = Zmax.reshape(-1, 1), Zmin.reshape(-1, 1)
    Zmean = Zmean.reshape(-1, 1)
    Z = (Z - Zmean) / (Zmax - Zmin)
    return Z


def Data_Reading(Normalization=True):
    train_x = np.load('F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\3\\trainset2.npy')
    train_y = np.load('F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\trainlabel2.npy')
    test_x = np.load('F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\3\\testset2.npy')
    test_y = np.load('F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\testlabel2.npy')

    # Normalization
    train_x = Normlize(train_x)
    test_x = Normlize(test_x)

    # xlsx to tensor
    if Normalization:
        train_x = torch.from_numpy(train_x).type(torch.cuda.FloatTensor)
        test_x = torch.from_numpy(test_x).type(torch.cuda.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.int64)
        test_y = torch.from_numpy(test_y).type(torch.int64)

    else:
        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.int64)
        test_y = torch.from_numpy(test_y).type(torch.int64)
    # reshape
    train_x = train_x.view(525, 10, 120)
    test_x = test_x.view(175, 10, 120)
    
    permutation = np.random.permutation(train_y.shape[0])
    train_x = train_x[permutation, :, :]
    print('---------------------------------------------------------------------------')
    train_y = train_y[permutation]

    return train_x, test_x, train_y, test_y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size=120, hidden_size=64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, 7)
        
    
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#cuda:0

    print(device)

    cnn = Net()
    print(cnn)
    cnn.to(device)

    lrr = 0.005
    #sgd -> stochastic gradient descent
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lrr)#
    loss_func = nn.CrossEntropyLoss()

    train_x, test_x, train_y, test_y = Data_Reading(Normalization=1)
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)

    te_x = Variable(test_x)
    te_y = Variable(test_y)

    batch_size = 25
    sum = 0
    sum_epo = 0
    for epoch in range(200):
        for i in range(0,(int)(len(train_x)/batch_size)):        
            t_x = Variable(train_x[i*batch_size:i*batch_size+batch_size].view(-1, 10, 120))         
            t_y = Variable(train_y[i*batch_size:i*batch_size+batch_size])

            output = cnn(t_x)                              
            loss = loss_func(output, t_y)                   
            optimizer.zero_grad()                  
            loss.backward()                               
            optimizer.step()                       

        test_output = cnn(test_x)               
        pred_y = torch.max(test_output, 1)[1]
        for j in range(test_y.size(0)):
            if pred_y[j] == test_y[j]:
                sum += 1
        accuracy = sum /test_y.size(0)
        sum = 0

        print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

        out = cnn(train_x)
        predicted_train = torch.max(out.data, 1)[1]
        total_train = train_y.size(0)
        for j in range(train_y.size(0)):
            if predicted_train[j] == train_y[j]:
                sum_epo += 1
        
        print('total_train:{}, accuracy:{}, sum:{}'.format(total_train, sum_epo / total_train, sum_epo))
        running_loss = 0
        sum_epo = 0

        if (sum / total_train > 0.85) :
            optimizer = torch.optim.Adam(cnn.parameters(), lr=lrr/8)
        elif (sum / total_train > 0.95) :
            optimizer = torch.optim.Adam(cnn.parameters(), lr=lrr/8/8)
        print('--------------------------------------------------------------------------')
    print('===========================================================================')
