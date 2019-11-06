'''
用于测试pytorch中卷积核反卷积
'''
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch

print('=========================================================')
conv = nn.Conv2d(1, 1, (2, 2), 1, 0, bias = False)
dconv2 = nn.ConvTranspose2d(in_channels = 1, out_channels = 1,  kernel_size = 2, stride = 1, padding = 0, output_padding = 0, bias = False)
input2 = Variable(torch.ones(1, 1, 4, 4))
init.xavier_uniform_(conv.weight)
init.xavier_uniform_(dconv2.weight)
print('conv.weight =', conv.weight)
print('dconv2.weight =', dconv2.weight)
print('------------------------------------------------')
print(input2)
print(conv(input2))
print(dconv2(conv(input2)))
#可以看到，这个反卷积会自己在周围先加边，然后把卷积核反过来，再相乘