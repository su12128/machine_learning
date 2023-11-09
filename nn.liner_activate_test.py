# 线性层
# vgg16网络模型为例子

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

class LineActivateTest(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.line_obj = torch.nn.Linear(196608,10)
    def forward(self,x):
        return self.line_obj(x)




def test():
    
    datastes = torchvision.datasets.CIFAR10('./datasets',train=False,transform=torchvision.transforms.ToTensor(),download=True)
    dataloader = DataLoader(dataset=datastes,batch_size=64,drop_last=True)
    writer = SummaryWriter('./logs')
    linear_obj = LineActivateTest()
    step =0
    for data in dataloader:
        imgs,trages = data
        print('-------------------')
        print(imgs.shape)
        writer.add_images('line_test_imgs',imgs,global_step=step)
        # 线性就是将前三个参数变为1，这里先看一下最后一个是多少，好确认Linear的参数
        # 转1维数组方式1
        # output = torch.reshape(imgs,(1,1,1,-1))
        # torch.flatten()展平，多维数组转1维
        # 转1维数组方式2
        output=torch.flatten(imgs)
        print(output.shape)#torch.Size([1, 1, 1, 196608])
        output=linear_obj(output)
        print(output.shape)
        # writer.add_images('line_test_output',output,global_step=step)
        step =step+1


if __name__ =='__main__':
    test()