"""
author: PWG
date: 2022-03-31 15:56
"""
##手写数字识别
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib. pyplot as plt
from matplotlib import cm
EPOCH=10;
LR=0.01
BATCH_SIZE=50
DOWNLOAD_MINST=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tarin_data=torchvision.datasets.MNIST(
    root='Data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINST
)
print(tarin_data.data.size())
print(tarin_data.targets.size())
# plt.imshow(tarin_data.data[10],cmap='gray')
# plt.title(' %i'%tarin_data.targets[10])
# plt.show()
train_loader=Data.DataLoader(
    dataset=tarin_data,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 是否打乱顺序
    num_workers=2,
)
test_data=torchvision.datasets.MNIST(root='Data/',train=False);
test_x = (torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].to(device))/255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:2000].to(device=device)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride = 1,
                padding=2,##padding=(kernel_size-1)/2
            ),#(16,28,28)
            nn.ReLU(),#(16,28,28)
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )#(16,14,14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),#(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)#(32,7,7)
        )
        self.out=nn.Linear(32*7*7,10)#表示0-9十个数

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平
        output = self.out(x)
        return output,x

try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

if __name__ == '__main__':
    cnn = CNN()
    cnn.cuda()
    # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    plt.ion()
# training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            output = cnn(b_x.to(device))[0]  # cnn output
            loss = loss_func(output, b_y.to(device))  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 10 == 0:
                test_output, last_layer = cnn(test_x)
                print(last_layer)
                pred_y = torch.max(test_output, 1)[1].cuda().data
                accuracy = float((pred_y == test_y.data).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(),
                      '| test accuracy: %.2f' % accuracy);
                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000,learning_rate='auto')
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
                    labels = test_y.cpu().numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
    plt.ioff()
    # print 10 predictions from test data
    test_output, _ = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10].cpu().numpy(), 'real number')


