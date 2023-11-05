# transforms.py中，Totensor 转化为tensor
# torchvision计算机视觉先关


from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def transform_test():
    img_path=''
    img=Image.open(img_path)

    writer = SummaryWriter('./logs')
    tensor_totensor = transforms.ToTensor()#将图片转化为tensor类型
    tensor_img = tensor_totensor(img)#tensor_img是一个tensor类型

    writer.add_image('tensor_test',tensor_img)
    writer.close()

def normalize_test():
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

