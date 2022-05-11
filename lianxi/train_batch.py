"""
author: PWG
date: 2022-03-30 15:48
"""
#批训练
import torch
import torch.utils.data as Data
BATCH_SiZE=10
x = torch.linspace(1, 10, 30)
y = torch.linspace(1,10, 30)
torch_dataset=Data.TensorDataset(x, y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SiZE,
    shuffle = True ,#是否打乱顺序
    num_workers=2,  #节点
)
def show_batch():
    for epoch in range(4):
        for step,(batch_x,batch_y) in enumerate(loader):
            #training
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())

if __name__ == '__main__':
    show_batch()




