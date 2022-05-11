"""
author: PWG
date: 2022-03-29 12:18
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib. pyplot as plt
##画常用的激活函数
# fake data
x = torch. linspace (-5, 5, 200) # x data (tensor), shape=(100, 1)
x=Variable(x)
x_np = x.data.numpy()
y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid (x).data.numpy()
y_tanh = torch.tanh (x). data. numpy ()
y_softplus = F.softplus (x). data. numpy()
plt.figure(1, figsize=(8, 6))
plt.subplot (221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 10))
plt.legend(loc='best')
plt.subplot (222)
plt.plot(x_np, y_sigmoid, c='red', label='sigşoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')
plt.subplot (223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='y_softplus')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
plt.show()


