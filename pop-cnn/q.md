1. 为什么```train_y = train_y / 10000    testy1 = testy1 / 10000     testy2 = testy2 / 10000```，就这些标签为什么要做这种处理？  A:归一化
2. ```init.xavier_uniform_(m.weight)```，和```init.normal_(m.weight, std=0.01)```是怎么选取的？
3. 这些loss分别怎么看    A:只要看train的loss
4. 如何打标签       A:没有特定格式
5. 全连接层的std = 0.01是什么意思？ 
6. 全连接层有bias吗？      A:有
7. optimizer.step()如何计算的？ A:loss计算了一个均值