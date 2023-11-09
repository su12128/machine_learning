# Sequential就是将多个模型整合并按顺序执行
from collections import OrderedDict
import torch 
model = torch.nn.Sequential(
          torch.nn.Conv2d(1,20,5),
          torch.nn.ReLU(),
          torch.nn.Conv2d(20,64,5),
          torch.nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = torch.nn.Sequential(OrderedDict([
          ('conv1', torch.nn.Conv2d(1,20,5)),
          ('relu1', torch.nn.ReLU()),
          ('conv2', torch.nn.Conv2d(20,64,5)),
          ('relu2', torch.nn.ReLU())
        ]))