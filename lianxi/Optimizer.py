"""
author: PWG
date: 2022-03-30 16:48
"""
#优化器对比
import torch
import matplotlib. pyplot as plt
import torch.utils.data as Data
import numpy as np
LR=0.01
BATCH_SIZE=40
EPOCH=12
x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
torch_dataset=Data.TensorDataset(x, y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle = True ,#是否打乱顺序
    num_workers=4,  #节点
)
def show_opt():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
    )
    net_SGD = net
    net_Momentum = net
    net_RMSprop = net
    net_ADam = net
    nets = [net_SGD, net_Momentum, net_RMSprop, net_ADam];
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momenum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_ADam = torch.optim.Adam(net_ADam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momenum, opt_RMSprop, opt_ADam]
    loss_func = torch.nn.MSELoss()
    loss_his = [[], [], [], []]
    for epoch in range(EPOCH):
        print(epoch)
        for step, (b_x, b_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, optimizers, loss_his):
                output = net(b_x)
                loss = loss_func(output, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())
            print(np.shape(loss_his))
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(loss_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()
if __name__ == '__main__':
    show_opt()