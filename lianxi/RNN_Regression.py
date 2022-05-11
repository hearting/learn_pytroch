"""
author: PWG
date: 2022-04-01 20:24
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import numpy as np
LR=0.01
TIME_STEP=20
INPUT_SIZE=1
steps=np.linspace(0,np.pi*2,100,dtype=np.float32)
x_np=np.sin(steps)
y_np=np.cos(steps)
plt.plot(steps,y_np,'r-',label='target(cos)')
plt.plot(steps,x_np,'b-',label='target(sin)')
plt.legend(loc='best')
plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=3,
            batch_first=True# input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out=nn.Linear(32,1)#全连接层
    def forward(self, x,h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)  # (batch,time step,input_size)
        outs=[]
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

rnn=RNN()
print(rnn)
optimizer=torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func=nn.MSELoss()
h_state=None
plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot
for step in range(60):
    start,end=step*np.pi,(step + 1)*np.pi
    steps=np.linspace(start,end,TIME_STEP,dtype=np.float32)
    x_np=np.sin(steps)
    y_np = np.cos(steps)
    x=torch.from_numpy(x_np[np.newaxis,:,np.newaxis]) # shape (batch, time_step, input_size)
    y=torch.from_numpy(x_np[np.newaxis,:,np.newaxis])
    prediction,h_state=rnn(x,h_state)
    h_state=h_state.data
    loss = loss_func(prediction, y)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.pause(0.05)

plt.ioff()
plt.show()
