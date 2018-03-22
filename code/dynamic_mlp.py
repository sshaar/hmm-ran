import torch.nn as nn
import torch

import os

import numpy as np

import os
import math
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, x) tensors to (n*m, x).
    """
    def forward(self, x):
        x = x.view(x.size()[0] * x.size()[1], x.size()[2])
        return x

class DynamicMLP(nn.Module):
    def __init__(self, input_size, output_size, num_layers=5, hidden_size=256):
        super(DynamicMLP, self).__init__()
        self.layers = []

        for i in range(num_layers):
            layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, dropout=0)
            self.layers.append(layer)

        self.decoder = nn.Linear(hidden_size, output_size)

        self.flatten = Flatten()

    def forward(self, input, layer_seq):

        for idx in range(input.shape[0]):

            time_step = input[idx]
            time_step = time_step.view((1, 1, 40))

            layer = self.layers[layer_seq[idx]]

            if idx == 0:
                output, (h_i, c_i) = layer(time_step)
            else:
                output, (h_i, c_i) = layer(time_step, (h_i, c_i))

        return self.flatten(self.decoder(output))

data = np.load('train.npz')

feat = data['feat']
label = data['target']
label_h = np.zeros(label.shape, dtype=int)

for i in range(label.shape[0]):
    if i < label.shape[0]/2:
        label_h[i] = 0
    else:
        label_h[i] = 1


feat = Variable(torch.from_numpy(feat), requires_grad=True).float()
label = Variable(torch.from_numpy(label_h)).long()
print label

# audio = feat[0]

dd = DynamicMLP(40, 2)
# output = dd(audio, [1]*130)
# print(output.grad)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(dd.parameters(), lr=1e-4, momentum=0.9)
dd.train()
for i in range(feat.shape[0]):
    audio = feat[i]

    output_audio = dd(audio, [1]*130)

    loss = criterion(output_audio, label[i])
    loss.backward()

    optimizer.step()

    if (i+1) % 10 == 0:
        optimizer.zero_grad()

    print i, loss.data[0]



# print output.shape
# print label[0].shape
# loss = criterion(output, label[0])
# loss.backward()
# print(loss.grad)
# print loss.data[0]
# optimizer.step()
