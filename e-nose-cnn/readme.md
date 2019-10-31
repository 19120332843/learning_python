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

## 2019.10.29

之前的实验不理想，现在重新设计实验。

实验原则：

1. 把同一个标号的数据求均值，做一个样本；

|名字|数据|
|:-:|:-:|
|槟榔|18+15+15=48|
|高良姜|20|
|砂仁|31|
|枳壳|100+|
|莪术|27|
|干姜|25+25+25+25=100|
|牡丹皮|8+27+27+27+27=116|

2. 实验数据加入新东西；

### 解决问题

分类类别不确定，每类样本数量不确定的分类问题。

### 首先进行数据处理

1. 以实验原则为底线，处理数据；

2. 在代码中加入新的想法。

### 参考论文

Cost-sensitive learning of deep feature representations from imbalanced data. 
