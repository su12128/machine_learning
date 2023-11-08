# 一个卷积就是一个特征，卷积就是去除干扰
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# train 是否使用训练数据（大）
# 数据集准备
datasets = torchvision.datasets.CIFAR10(
    './datasets', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 加载数据
dataloader = DataLoader(datasets, batch_size=64)


class Conv2dTest(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 彩图channel是3 所以 in_channels=3
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


def conv_test():
    conv_obj = Conv2dTest()
    # print(conv_obj)
    writer = SummaryWriter('./logs')
    step = 0
    for data in dataloader:
        imgs, targes = data
        output = conv_obj(imgs)
        print(imgs.shape)#[64, 3, 32, 32]
        print(output.shape)#torch.Size([64, 6, 30, 30])
        writer.add_images('conv2d_test_images', imgs, global_step=step)
        # 通道为6无法显示，报错，size of input tensor and input format are different.         tensor shape: (64, 3, 32, 32), input_format: CHW
        # 要将[16, 6, 30, 30] 改为 [x, 3, 30, 30] 使用torch.reshape()
        # 使用-1的话就是自动计算
        output = torch.reshape(output,(-1,3,30,30))
        writer.add_images('conv2d_test_output', output, global_step=step)
        step = step+1

    writer.close()


if __name__ == '__main__':
    conv_test()
