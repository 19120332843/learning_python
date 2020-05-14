import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv1 = ["codedata/3times/csv/binglang-0.csv", "codedata/3times/csv/binglang-1.csv", "codedata/3times/csv/binglang-2.csv"]

def Normlize(Z):
    Zmax, Zmin = Z.max(axis=0), Z.min(axis=0)
    Zmean = Z.mean(axis=0)
    Z = (Z - Zmean) / (Zmax - Zmin)
    return Z


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

file_paths = []
listdir("F:\\gitworkspace\\python\\e-nose-cnn\\codedata\\3times\\csv", file_paths)
# print(file_paths)

for i in csv1:
    pic = pd.read_csv(i)
    # pic1 = Normlize(pic)

    x = np.arange(0, 120, 1)

    y0 = pic['0']
    y1 = pic['1']
    y2 = pic['2']
    y3 = pic['3']
    y4 = pic['4']
    y5 = pic['5']
    y6 = pic['6']
    y7 = pic['7']
    y8 = pic['8']
    y9 = pic['9']

    plt.xlim(-1, 130)
    plt.ylim(0, pic.max().max() + 2)

    plt.plot(x, y0, 'yellow')
    plt.plot(x, y1, 'red')
    plt.plot(x, y2, 'blue')
    plt.plot(x, y3, 'chocolate')
    plt.plot(x, y4, 'crimson')
    plt.plot(x, y5, 'dimgray')
    plt.plot(x, y6, 'forestgreen')
    plt.plot(x, y7, 'green')
    plt.plot(x, y8, 'maroon')
    plt.plot(x, y9, 'orange')

    plt.show()
    # print(i)

    # y0 = pic1['0']
    # y1 = pic1['1']
    # y2 = pic1['2']
    # y3 = pic1['3']
    # y4 = pic1['4']
    # y5 = pic1['5']
    # y6 = pic1['6']
    # y7 = pic1['7']
    # y8 = pic1['8']
    # y9 = pic1['9']

    # plt.xlim(-1, 130)
    # plt.ylim(pic1.min().min() - 2, pic1.max().max() + 2)

    # plt.plot(x, y0, 'yellow')
    # plt.plot(x, y1, 'red')
    # plt.plot(x, y2, 'blue')
    # plt.plot(x, y3, 'chocolate')
    # plt.plot(x, y4, 'crimson')
    # plt.plot(x, y5, 'dimgray')
    # plt.plot(x, y6, 'forestgreen')
    # plt.plot(x, y7, 'green')
    # plt.plot(x, y8, 'maroon')
    # plt.plot(x, y9, 'orange')

    # plt.show()
    #plt.savefig("F:\\gitworkspace\\python\\e-nose-cnn\\draw\\train\\1\\")
