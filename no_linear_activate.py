# 非线性激活
# RELU：0：x<0;x:x.0
# Sigmoid
import torch

class ReluTest(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forwoard(self,x):
        reul_obj = torch.nn.ReLU(inplace=True)
        return reul_obj(x)


def test_relu():
    # inplace 是否用结果替换输入，一般给False
    reul_obj = torch.nn.ReLU(inplace=True)
    # input = torch.randn(2)
    input = torch.tensor([[1,2],[-1,-2]])
    input = torch.reshape(input,(-1,1,2,2))
    output = reul_obj(input)
    # 因为inplace =True 所以这里input的值会被设置为与outout一致
    print(output)



if __name__ =='__main__':
    test_relu()