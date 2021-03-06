"""
author: PWG
date: 2022-03-30 14:07
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib. pyplot as plt
n_data=torch.ones(100,2)
x0 = torch. normal(2*n_data, 1)# classe x data (tensor), shape=(100, 2)
y0= torch. zeros(100)# classe y data (tensor), shape=(100, 1)
x1 = torch. normal(-2*n_data, 1)# class1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)# class1 y data (tensor), shape=(100, 1)
x = torch. cat((x0, x1), 0). type(torch. FloatTensor) # FloatTensor = 32-bit floating y = torch. cat((yo, y1), ). type(torch. LongTensor)
y=torch.cat((y0,y1),0).type(torch.LongTensor)
x,y=Variable(x),Variable(y)
plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap=None)
plt.show()
#方法一
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
    def forward(self, x):#前向传播
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x;
net=Net(2,10,2);
print(net)
plt.ion()#画动态图
# 优化
optimizer=torch.optim.SGD(net.parameters(), lr=0.01)
loss_func=torch.nn.CrossEntropyLoss()#标签误差
for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out,1), 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 15, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
## 方法二
net2=torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
)
print(net2)


