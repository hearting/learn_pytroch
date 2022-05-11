"""
author: PWG
date: 2022-03-31 23:12
"""
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

EPOCH=10;
LR=0.01
BATCH_SIZE=50
TIME_STEP=28
INPUT_SIZE=28
DOWNLOAD_MINST=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tarin_data=torchvision.datasets.MNIST(
    root='Data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINST
)
train_loader=Data.DataLoader(
    dataset=tarin_data,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 是否打乱顺序
    num_workers=2,
)
test_data=torchvision.datasets.MNIST(root='Data/',train=False);
test_x = (test_data.data.type(torch.FloatTensor)[:2000]/255).to(device)  # shape from (2000, 28, 28) , value in range(0,1)
print(test_x.size())
test_y = test_data.targets[:2000].numpy()
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers = 2,
            batch_first=True,# input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out=nn.Linear(64,10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
       r_out,(h_n,h_c)=self.rnn(x,None)#(batch,time step,input_size)
       output=self.out(r_out[:,-1,:])#选择最后时刻的output
       return output
if __name__ == '__main__':
    rnn=RNN().to(device)
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
            b_x = b_x.to(device).view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)

            output = rnn(b_x.to(device))  # rnn output
            loss = loss_func(output, b_y.to(device))  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                test_output = rnn(test_x)  # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                accuracy = float(sum(pred_y == test_y)) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

    # print 10 predictions from test data
    test_output = rnn(test_x[40:50].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    print(pred_y, 'prediction number')
    print(test_y[40:50], 'real number')

