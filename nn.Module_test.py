# 神经网络的基本骨架nn.Moudle
# nn Neural network 神经网络

# Convolution Layers 卷积层
# Pooling layers 池化层
# Padding Layers 填充层
# Non-linear Activations (weighted sum, nonlinearity) 非线性激活（加权和、非线性）

# 所有的module 都要继承torch.nn.Module这一父类
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn
import torch

class MyModule(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1= nn.Conv1d(1,20,5)#卷积
        self.conv2=nn.Conv2d=(20,20,5)

    def forward(self,x):
        #所有继承torch.nn.Module的子类都应该重写这一方法
        # 前进函数
        # 前向传播
        #F.relu 非线性处理  激活函数
        # relu 是分段函数：0，x<0;x,x>0
        x=F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
    
    # 输入x->卷积->非线性->卷积->非线性->输出

class TestModule(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,x):
        return x+1

def contarinter_test():
    input = torch.tensor(1.0)
    test_module = TestModule()

    out_put=test_module(input)

    print(out_put)





if __name__ =='__main__':
    contarinter_test()