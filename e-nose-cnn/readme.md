# e-nose cnn

把以下四种数据放入pop-cnn，测试分类效果。

binglang-槟榔

gaoliangjiang-高良姜

sharen-砂仁

zhike-枳壳

## draw_test

画图测试

## nos-data

原始数据，其中，每个文件夹的1是训练集，2是测试集

## train

训练集

## test

测试集

## 数据整理流程

1. 把每个nos转为csv,一定要加header；

2. 把每个csv转置并合成一个csv，一定要加header；

## pop-cnn

在原来的基础上，改成分类

1. 先把csv转成numpy想要的格式npy；
