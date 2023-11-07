# Dataloader 的使用
# 数据加载器，将数据加载在神经网络中，在dataloader中设置如何取datasets中的数据，如何取，取多少等
import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def dataloader_test():

    test_data=torchvision.datasets.CIFAR10('./datasets',train=False,transform=torchvision.transforms.ToTensor())
    # 参数：dataset：数据集；batch_size：每次装载多少样本；shuffle：是否打乱；
    # drop_last：取余后多出的余数数据是否丢弃；num_workers：设置子进程数量，默认为0在主进程执行
    # sampler 采样器默认是RandomSample 随机采样，随机抓取
    
    dataloader_obj = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
    # 测试集中第一张样本图
    img,targe=test_data[0]
    print(img.shape)
    print(targe)

    # DataLoader就会将数据集中的样本进行打包，batch_size=4,那么数据集中每4张图就进行打包得到args[]和targe[],其中长度就为batch_size
    # 返回的是一个迭代器，其中每个iter中都有args和targes
    writer=SummaryWriter('./logs')
    for epoch in range(2):#0,1  读取两遍,这时候shuffle为True，那么就会进行洗牌，两次的结果就会不一致
        step=0
        for data in dataloader_obj:
            imgs,targes=data
            
            print(imgs.shape)#torch.Size([4, 3, 32, 32]) 4张图片，3通道，宽高是32*32
            # 这里的images就是神经网络的一个输入
            print(targes)#tensor([8, 8, 9, 7])4张图片的targe
            writer.add_images(f'Dataloader_test{epoch}',imgs,global_step=step,)
            step=step +1

    writer.close()




if __name__ == '__main__':
    dataloader_test()