from typing import Any
from torch.utils.data import Dataset
import torch
import cv2
import os
from PIL import Image

#继承Dataset需要重写getitem 和len方法啊
# MyData主要用作将训练的数据进行集合，作为输入
class MyData(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self):
        pass


if __name__ == "__main__":
    print(torch.cuda.is_available())
