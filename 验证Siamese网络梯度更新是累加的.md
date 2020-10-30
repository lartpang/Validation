# 验证Siamese网络梯度更新是累加的

这是知乎上一个问题的回答，这里存放下：Siamese网络是如何更新权重的？ - 人民艺术家的回答 - 知乎 <https://www.zhihu.com/question/372743063/answer/1551338839>

```python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/30
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @GitHub  : https://github.com/lartpang

# 参考https://zhuanlan.zhihu.com/p/75054200
import copy
import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import SGD


def initialize_seed_cudnn(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 可以影响shuffle的结果
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 确保随机数固定，每次运行都一样
initialize_seed_cudnn(0)

total_grad_out = []
total_grad_in = []


# 针对模块的钩子
def hook_fn_backward(module: nn.Module, grad_input: torch.Tensor,
                     grad_output: torch.Tensor):
    print(module)  # 为了区分模块
    # 为了符合反向传播的顺序，我们先打印 grad_output
    print('model_grad_out', len(grad_output), grad_output[0].mean())
    # 再打印 grad_input
    # print('model_grad_in', len(grad_input), [x.shape for x in grad_input])
    # 保存到全局变量
    total_grad_in.append(grad_input)
    total_grad_out.append(grad_output)


# 针对tensor的钩子
def hook_fn_inter_tensor(grad):
    print("jpeg's grad:")
    print(grad.mean())


jpeg = torch.ones(1, 1, 40, 40, requires_grad=True).cuda()

# 注册tensor钩子
jpeg.register_hook(hook=hook_fn_inter_tensor)


class TestRandomNumEleMul(nn.Module):
    def __init__(self):
        super(TestRandomNumEleMul, self).__init__()
        self.coef = nn.Parameter(torch.randn(1, 1, 40, 40))

    def forward(self, x):
        return self.coef * x


model_1 = TestRandomNumEleMul().cuda()
model_2 = TestRandomNumEleMul().cuda()

# 确保是相同的参数但不是同一个模块，直接等于是不行的，共享相同地址
model_2.coef = copy.deepcopy(model_1.coef)
print(model_2.coef.storage().data_ptr())
print(model_1.coef.storage().data_ptr())
# 140319184145920
# 140319184132608

# 注册模块钩子
model_1.register_backward_hook(hook_fn_backward)
model_2.register_backward_hook(hook_fn_backward)

# 只是为了便于区分模块，起个名字
net = nn.ModuleDict(dict(model1=model_1, model2=model_2))

# 使用最简单的SGD优化器
optimizer = SGD(params=net.parameters(), lr=1)
optimizer.zero_grad()

# 模拟Siamese结构
print("模拟Siamese结构")
# jpeg's grad:
# tensor(6.8778e-05, device='cuda:0')
a = net['model1'](jpeg)
# TestRandomNumEleMul()
# model_grad_out 1 tensor(0.0006, device='cuda:0')
b = net['model1'](jpeg)
# TestRandomNumEleMul()
# model_grad_out 1 tensor(0.0012, device='cuda:0')

c = a + 2 * b
loss = c.mean()
loss.backward()

print('model1_before_update\n',
      net['model1'].coef.mean(),
      net['model1'].coef.max(),
      net['model1'].coef.min(), sep='\n')
print('model2_before_update\n',
      net['model2'].coef.mean(),
      net['model2'].coef.max(),
      net['model2'].coef.min(), sep='\n')
# model1_before_update
#
# tensor(0.0367, device='cuda:0', grad_fn=<MeanBackward0>)
# tensor(4.1015, device='cuda:0', grad_fn=<MaxBackward1>)
# tensor(-3.1537, device='cuda:0', grad_fn=<MinBackward1>)
# model2_before_update
#
# tensor(0.0367, device='cuda:0', grad_fn=<MeanBackward0>)
# tensor(4.1015, device='cuda:0', grad_fn=<MaxBackward1>)
# tensor(-3.1537, device='cuda:0', grad_fn=<MinBackward1>)

# 更新参数
optimizer.step()

print('model1_after_update\n',
      net['model1'].coef.mean(),
      net['model1'].coef.max(),
      net['model1'].coef.min(), sep='\n')
print('model2_after_update\n',
      net['model2'].coef.mean(),
      net['model2'].coef.max(),
      net['model2'].coef.min(), sep='\n')
# model1_after_update
#
# tensor(0.0348, device='cuda:0', grad_fn=<MeanBackward0>)
# tensor(4.0996, device='cuda:0', grad_fn=<MaxBackward1>)
# tensor(-3.1556, device='cuda:0', grad_fn=<MinBackward1>)
# model2_after_update
#
# tensor(0.0367, device='cuda:0', grad_fn=<MeanBackward0>)
# tensor(4.1015, device='cuda:0', grad_fn=<MaxBackward1>)
# tensor(-3.1537, device='cuda:0', grad_fn=<MinBackward1>)

# new_param = old_param - lr * grad
# mean, max, min 也满足
# 0.0348 = 0.0367 - (0.0006 + 0.0012)
# 4.0996 = 4.1015 - (0.0006 + 0.0012)
# -3.1556 = -3.1537 - (0.0006 + 0.0012)
```
