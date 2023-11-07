# torchvision datasets 数据集的使用
import torchvision


def torchvision_test():
    '''
    文档位置：https://pytorch.org/docs/stable/index.html
    datasets文档位置：https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10
    就是数据集位置
    '''
    # torchvision 中的datasets 就是数据集，可以选择下载
    # 参数1：root 数据集存放位置，download=True 表示如果root目录下没有就去从网上下载
    trans_compose =torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])
    train_CIFAR10_set=torchvision.datasets.CIFAR10('./datasets',train=True ,download=True,transform=trans_compose)
    test_CIFAR10_set=torchvision.datasets.CIFAR10('./datasets',train=False ,download=True,transform=trans_compose)
    # 数据集返回的是img,targe
    print(test_CIFAR10_set[0])
    print(test_CIFAR10_set.classes)
    img,targe= test_CIFAR10_set[0]
    print(img.shape)#图片格式信息[C,H,W]
    print(f'targe{targe}')

    print(test_CIFAR10_set.classes[targe])
    # img.show()

if __name__ == '__main__':
    torchvision_test()