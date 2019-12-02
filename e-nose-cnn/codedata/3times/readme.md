# 数据说明

1. csv文件夹里放的是所有数据，一共7类，每类100个样本，编号从0-99。每个样本是120行10列（意思是120秒采集，每秒采集一次，一共10个传感器）；

2. `data.csv`是csv文件夹里所有样本的集合到一个csv文件中，每个样本是10行120列（注意，这里和原本120行10列不同，进行转置了），所以这个文件一共有7001行，多出来的一行是标题；

3. `label.xlsx`是标签；

4. 两个`*.npy`是numpy格式的文件，`dataset.npy`由`data.csv`转的，`label.npy`由`label.xlsx`转的，具体怎么转的可以看`\learning_python\e-nose-cnn\datareading.py`；
