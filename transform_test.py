# transforms.py中，Totensor 转化为tensor
# torchvision计算机视觉先关


from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def transform_test():
    '''
    图片转tensor
    '''
    img_path=''
    img=Image.open(img_path)

    writer = SummaryWriter('./logs')
    # 输入PIL，输出tensor
    tensor_totensor = transforms.ToTensor()#将图片转化为tensor类型
    tensor_img = tensor_totensor(img)#tensor_img是一个tensor类型

    writer.add_image('tensor_test',tensor_img)
    writer.close()

def normalize_test():
    '''
        归一化处理
    '''
    img_path = ''
    img_pil=Image.open(img_path)
    writer = SummaryWriter('./logs')
    tensor_totensor = transforms.ToTensor()  # 将图片转化为tensor类型
    tensor_img = tensor_totensor(img_pil)  # tensor_img是一个tensor类型

    print(tensor_img[0][0][0])
    trans_normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])#用平均值和标准偏差归一化张量图像。只支持输入tensor
    #归一化公式：output[channel] = (input[channel] - mean[channel]) / std[channel]
    #归一化是让数据在一个范围内，从而避免奇异样本数据的影响
    img_nor=trans_normalize(tensor_img)
    print(img_nor[0][0][0])

    writer.add_image('noramlize_test',img_nor)

def img_resize():
    '''
    图片缩放
    '''
    img_path=''
    img_pil=Image.open(img_path)
    trans_resize = transforms.Resize(512)
    # 随机裁剪
    trans_radom = transforms.RandomCrop(512)
    # 输入PIL 输出也是PIL
    resize_img = trans_resize(img_pil)
    # 之后就是转tensor
    tensor_totensor = transforms.ToTensor()

    resize_img = tensor_totensor(resize_img)


def compose_test():
    '''
    Compose中的参数是一个列表
    数据需要是transofroms类型的对象
    相当于就是Compose将操作整合，前面的输出会作为后面的输入（这里要注意前面输出的数据类型是否是后面正确的输入类型）
    '''
    img_path=''
    img=Image.open(img_path)
    trans_resiz=transforms.Resize(512)
    trans_totensor=transforms.ToTensor()
    # 实例化transforms.Compose
    trans_compose=transforms.Compose([trans_resiz,trans_totensor])
    img_tensor = trans_compose(img)
    # 结果查看
    writer = SummaryWriter('./logs')
    # 参数1：tage;参数2：tensor格式的图片数据；参数3：步数
    writer.add_image('Image resize and to tensor',img_tensor,0)

    writer.close()
