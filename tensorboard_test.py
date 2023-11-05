# 记录训练、验证过程，并对数据进行可视化
# 先运行writer.add_scalar(),构造好logs目录下的数据文件
# 方式1：在终端命令行中输入命令：tensorboard --logdir=logs --port=6006
# 需要安装tensorboard: pip install tensorboard
# 处于同一目录下（logs，且是同一个tillter-参数1）的数据会进行线性拟合（这就是训练），不同的数据集要根据使用不同的目录（模型目录）
import os

import numpy
from PIL import Image

from torch.utils.tensorboard import SummaryWriter


# writer.add_image()


# writer.add_scalar()
# global_step  : X轴
# scalar_value  :Y轴
def test():
    # 参数logs可以理解为是模型目录
    writer = SummaryWriter('./logs')
    for i in range(0, 100):
        # 参数1：函数表达式-tillter-类别，参数2：y值，参数3：x值
        writer.add_scalar('y=2x', 2 * i, i)
    writer.close()


def test_read_img():
    writer = SummaryWriter('./img_logs')
    # writer.add_image()
    img_path = ''
    img_array = pil_img2numpy_array(img_path)
    writer.add_image('array_test', img_array, 1)
    # 参数1：tage;参数3：图片数据，参数3：步数

    # tag(str): Data identifier
    # img_tensor(torch.Tensor, numpy.ndarray, or string / blobname): Image
    # 一般使用OpenCV将图片转换为numpy型
    # 利用numpy.array() 对PIL类型图片进行转换
    # 输出图片格式  print(img.shape) 一般都是  通道、高、宽  （C，H，W）
    # numpy类型要使用dataformats='HWC'参数，就是需要指定numpy参数数数的数字\维度的含义  ！！！！


def pil_img2numpy_array(img_path):
    img_array = None
    if os.path.exists(img_path):
        img_pil = Image(img_path)
        img_array = numpy.array(img_pil)
    else:
        print('图片不存在，请检查图片路径')
    return img_array


if __name__ == '__main__':
    test()
