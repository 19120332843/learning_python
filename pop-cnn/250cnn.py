import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable


def Normlize(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Zmean = Z.mean(axis=1)
    #按列排序
    Zmax, Zmin = Zmax.reshape(-1, 1), Zmin.reshape(-1, 1)
    Zmean = Zmean.reshape(-1, 1)
    Z = (Z - Zmean) / (Zmax - Zmin)
    return Z

def Data_Reading(Normalization=True):
    # Read the xlsx
    '''
    //读取excel的值
    train_x = pd.read_excel( "trainingset.xlsx", 'Input',header = None)
    test_x = pd.read_excel("oilsset.xlsx",'oils',header = None)
    test_z = pd.read_excel("newodorset.xlsx",'new',header = None)
    train_y = pd.read_excel("trainingy.xlsx",'Output',header = None)
    testy1 = pd.read_excel("oilsy.xlsx",'oy',header = None)
    testy2 = pd.read_excel("newy.xlsx",'ny',header = None)
    
    //使用前一个观察值填充 
    train_x = train_x.fillna(method='ffill')
    test_x = test_x.fillna(method='ffill')
    text_z = test_z.fillna(method='ffill')
    train_y = train_y.fillna(method='ffill')

    np.save('trainingset.npy', train_x)
    np.save('oilsset.npy',test_x)
    np.save('newodorset.npy', test_z)
    np.save('trainingy.npy', train_y)
    np.save('testy1.npy', testy1)
    np.save('testy2.npy', testy2)
    '''
    train_x = np.load('F:\\gitworkspace\\python\\pop-cnn\\trainingset.npy')
    test_x = np.load('F:\\gitworkspace\\python\\pop-cnn\\oilsset.npy')
    test_z = np.load('F:\\gitworkspace\\python\\pop-cnn\\newodorset.npy')
    train_y = np.load('F:\\gitworkspace\\python\\pop-cnn\\trainingy.npy')
    testy1 = np.load("F:\\gitworkspace\\python\\pop-cnn\\testy1.npy")
    testy2 = np.load("F:\\gitworkspace\\python\\pop-cnn\\testy2.npy")
 

    # Normalization
    train_x_Normed = Normlize(train_x)
    test_x_Normed = Normlize(test_x)
    test_z_Normed = Normlize(test_z)
    train_y = train_y / 10000
    testy1 = testy1 / 10000
    testy2 = testy2 / 10000

    # xlsx to tensor
    if Normalization:
        train_x = torch.from_numpy(train_x_Normed).type(torch.cuda.FloatTensor)
        test_x = torch.from_numpy(test_x_Normed).type(torch.cuda.FloatTensor)
        test_z = torch.from_numpy(test_z_Normed).type(torch.cuda.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.cuda.FloatTensor)
        testy1 = torch.from_numpy(testy1).type(torch.cuda.FloatTensor)
        testy2 = torch.from_numpy(testy2).type(torch.cuda.FloatTensor)

    else:
        train_x = torch.from_numpy(train_x).type(torch.cuda.FloatTensor)
        test_x = torch.from_numpy(test_x).type(torch.cuda.FloatTensor)
        test_z = torch.from_numpy(test_z).type(torch.cuda.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.cuda.FloatTensor)
        testy1 = torch.from_numpy(testy1).type(torch.cuda.FloatTensor)
        testy2 = torch.from_numpy(testy2).type(torch.cuda.FloatTensor)


    # reshape
    train_x = train_x.view(238, 1, 16, 250)
    test_x = test_x.view(108, 1, 16, 250)
    test_z = test_z.view(95, 1, 16, 250)
    return train_x, test_x, test_z, train_y,testy1,testy2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,(16,4),stride=(1,3))
        self.conv2 = nn.Conv2d(6,10,(1,3),stride=(1,2))
        #self.conv3 = nn.Conv2d(10,14,(1,4),stride=(1,2))
        self.fc = nn.Linear(10*1*41,1)
        


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)


             
    def forward(self, x):
        x = F.relu(self.conv1(x))#（238,6,1,124）
        x = F.relu(self.conv2(x))#（238,16,1,23）
        #x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten the tensor
        x = self.fc(x)
        return x

#training
batch_size = 14
def train(train_x,train_y,step=20):
    for epoch in range(160):
        for i in range(0,(int)(len(train_x)/batch_size)):
            t_x = Variable(train_x[i*batch_size:i*batch_size+batch_size])
            t_y = Variable(train_y[i*batch_size:i*batch_size+batch_size])
            t_x = t_x.to(device)
            t_y = t_y.to(device)
            out = cnn(t_x)
            #forward
            #loss_func = nn.MSELoss() 均方损失函数  loss(x(i),y(i)) = (x(i) - y(i))^2
            loss = loss_func(out, t_y)
            #梯度初始化为零
            optimizer.zero_grad()
             
            #backward
            loss.backward()
            optimizer.step()
        if (epoch + 1) % step == 0:
            print('Epoch[{}/{}], loss: {:.12f},'.format(epoch + 1,160, loss.item()))
        
            
#predicting
def predict(test_x,testy1):
    for epoch in range(30):
        te_x = Variable(test_x)
        tey1 = Variable(testy1)
        out1 = cnn(te_x)
        loss1 = loss_func(out1, tey1)
        out1 = out1 * 10000
        #print('Epoch[{}/{}], loss1: {:.12f},'.format(epoch + 1, 30, loss1.item()))
        print('epoch, loss1:, out1', epoch + 1, loss1, out1)

def newodor(test_z,testy2):
    for epoch in range(10):
        te_z = Variable(test_z)
        tey2 = Variable(testy2)
        out2 = cnn(te_z)
        loss2 = loss_func(out2, tey2)
        out2 = out2 * 10000
        #print('Epoch[{}/{}], loss2: {:.12f},'.format(epoch + 1, 10, loss2.item()))
        print('epoch, loss2:, out2', epoch + 1, loss2, out2)




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    cnn = Net()
    print(cnn)
    cnn.to(device)

    #sgd -> stochastic gradient descent
    optimizer = optim.SGD(cnn.parameters(), lr=0.0001, momentum=0.8)
    loss_func = nn.MSELoss()

    train_x, test_x, test_z, train_y,testy1,testy2 = Data_Reading(Normalization=True)
    train(train_x, train_y, step=20)
    predict(test_x,testy1)
    newodor(test_z,testy2)

