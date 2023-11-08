# 最大池化 Maxpol（下采样）Maxunpool(上采样)
# stride 默认值是卷积核size 
# dilation 空洞卷积层，卷积核 矩阵值之间会有间隔
# ceil 向上取整 （保留卷积核移动后不够中部分）， floor 向下取整
# 池化核
# 最大池化就是取池化核 做运算中最大得一个输出，（卷积是每个对应相乘的和做输出）
# #池化函数使用某一位置的相邻输出的总体统计特征来代替网络在该位置的输出。本质是降采样，可以大幅减少网络的参数
import torch

# "max_pool2d" not implemented for 'Long'
# 最大池化接受数据喂浮点型，整型会报错
# 解决方法
# dtype=torch.float
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float)

input = torch.reshape(input,(-1,1,5,5))
print(input)

class MaxPoolTest(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxpool2d=torch.nn.MaxPool2d(kernel_size=3,ceil_mode=True)
    def forward(self,x):
        output = self.maxpool2d(x)
        return output
    
if __name__ =="__main__":

    maxpost_obj = MaxPoolTest()

    out_put=maxpost_obj(input)
    print(out_put)