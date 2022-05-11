"""
author: PWG
date: 2022-04-03 20:51
"""
##使用非监督学习自编码进行手写数字识别
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib. pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
EPOCH=40;
LR=0.001
BATCH_SIZE=50
DOWNLOAD_MINST=False
N_TEST_IMG =5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data=torchvision.datasets.MNIST(
    root='Data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINST
)
print(train_data.data.size())
print(train_data.targets.size())
# plt.imshow(train_data.data[10],cmap='gray')
# plt.title(' %i'%train_data.targets[10])
# plt.show()
train_loader=Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 是否打乱顺序
    num_workers=3,
)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 3),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

if __name__ == '__main__':
    autoencoder = AutoEncoder().to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()  # continuously plot

    # original data (first row) for viewing
    view_data = (train_data.data[5+N_TEST_IMG:15].view(-1, 28 * 28).type(torch.FloatTensor) / 255.).to(device)
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.cpu()[i], (28, 28)), cmap='gray');
        a[0][i].set_xticks(());
        a[0][i].set_yticks(())

    for epoch in range(EPOCH):
        t1=time.perf_counter()
        for step, (x, b_label) in enumerate(train_loader):
            b_x = x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
            b_y = x.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)

            encoded, decoded = autoencoder(b_x.to(device))

            loss = loss_func(decoded, b_y.to(device))  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu())

                # plotting decoded image (second row)
                _, decoded_data = autoencoder(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data.data.cpu()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(());
                    a[1][i].set_yticks(())
                plt.draw();
                plt.pause(0.05)
        t2=time.perf_counter()
        print("ecpoch:",epoch,"用时(s)：",t2-t1)

    plt.ioff()
    plt.show()
    pred_y = torch.max(view_data, 1)[1].data.cpu()
    print(pred_y, 'prediction number')
    # visualize in 3D plot
    view_data = (train_data.data[:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255.).to(device)
    encoded_data, _ = autoencoder(view_data)
    fig = plt.figure(2);
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    X, Y, Z = encoded_data.data[:, 0].cpu().numpy(), encoded_data.data[:, 1].cpu().numpy(), encoded_data.data[:, 2].cpu().numpy()
    values = train_data.targets[:200].numpy()
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255 * s / 9));
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max());
    ax.set_ylim(Y.min(), Y.max());
    ax.set_zlim(Z.min(), Z.max())
    plt.show()