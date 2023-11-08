# 卷积
import torch
def convd_test():
    #  输入的二维矩阵
    input = torch.tensor(
         [[1,2,0,3,1],
          [0,1,2,3,1],
          [1,2,1,0,0],
          [5,2,3,1,1],
          [2,1,0,1,1]])
    print(input.shape)
    #卷积核
    kernel = torch.tensor([[1,2,1],
                           [0,1,0],
                           [2,1,0]])
    print(kernel.shape)
    
    # stride 内核移动步数，1就是卷积核在二维数据上移动一步
    #[1,2,0]        [1,2,1]
    #[0,1,2]        [0,1,0]
    #[1,2,1]        [2,1,0] 跟卷积核做计算： 1*1+2*2+1*0+0*0+1*1+0*2+2*1+1*2+0*1 = 10
    # 然后卷积核香左移动stride步 继续计算 得到x
    # 最后结果就是一个3*3的矩阵
    #[10,x,y]
    #[z,a,b]       
    #[c,d,e]        
    #如果步数是2 就得到一个2*2的矩阵
    #

    # 这里面conv2d(N,C,H,W)里面的四个是 N就是batch size也就是输入图片的数量，C就是通道数这只是一个二维张量所以通道为1，H就是高，W就是宽，所以是1 1 5 5
    # 5*5 二维矩阵，通道为1；灰度图用二维矩阵表示，通道为1，；彩色图用三维矩阵表示通道数为2
    # 尺寸变换
    input = torch.reshape(input,(1,1,5,5))
    kernel=torch.reshape(kernel,(1,1,3,3))

    output1=torch.nn.functional.conv2d(input,kernel,stride=1)
    print(output1)

    output2=torch.nn.functional.conv2d(input,kernel,stride=2)
    print(output2)


    # padding 会对输入图像进行填充
    # 基本就是周围补0，让卷积后大小不变
    # 目的是为了防止图像内的某些像素点被多次进行特征提取，导致某些点的作用过于放大了
    # 填充是为了更好的保留边缘特征，简单来说中间块会被卷积利用多次，而边缘块使用的少，会造成不均
    output3=torch.nn.functional.conv2d(input,kernel,stride=1,padding=1)
    print(output3)


    #out_channel 几个输出通道，就代表用几个卷积核尽心运算（神经元个数）

if __name__ =='__main__':
    convd_test()