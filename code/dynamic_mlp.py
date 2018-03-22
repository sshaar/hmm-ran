import os
import numpy as np
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ACCUM_GRAD = 10


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
        # All the layers in the states of the HMM.
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

class DataClass (Dataset):
    def __init__ (self, feats, label, seq):
        self.feats = feats
        self.labels = label
        self.seq = seq

    def __getitem__ (self, index):
        return (self.feats[index], self.labels[index], seq[index])

    def __len__(self):
        return self.feats.shape[0]

def my_collate(batch):
    feats = np.array([item[0] for item in batch])
    labels = np.array([item[1] for item in batch])
    seqs = np.array([item[2] for item in batch])
    return (feats, labels, seqs)

def change_label(x):
    if ("blues" in x):
        return 0
    elif ("classical" in x):
        return 1
    elif ("country" in x):
        return 2
    elif ("disco" in x):
        return 3
    elif ("hiphop" in x):
        return 4
    elif ("jazz" in x):
        return 5
    elif ("metal" in x):
        return 6
    elif ("pop" in x):
        return 7
    elif ("reggae" in x):
        return 8



def label_to_int(labels):
    print ("LABELS", labels)
    a = list (map (lambda x : change_label(x[0]), labels))
    print (a)
    return np.array(a)


def load_data(batch_size, shuffle):
    ## loads training, validation and testing data
    train_data = np.load("/home/sshaar/hmm-rnn/train.npz")
    train_seq = np.load("../../data/train_seqs3.npy")
    valid_data = np.load("/home/sshaar/hmm-rnn/dev.npz")
    valid_seq = np.load("../../data/valid_seqs3.npy")
    test_data = np.load("/home/sshaar/hmm-rnn/test.npz")
    test_seq = np.load("../../data/test_seqs3.npy")

    ## creates dataset for the training, validation, test data
    train_data = DataClass(train_data["feat"], label_to_int(train_data["target"]), train_seq)
    valid_data = DataClass(valid_data["feat"], label_to_int(valid_data["target"]), valid_seq)
    test_data = DataClass(test_data["feat"], label_to_int(test_data["target"]), test_seq)

    ## data loaders for training, validation and testing data
    train_loader = DataLoader(dataset = train_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)
    valid_loader = DataLoader(dataset = valid_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)
    test_loader = DataLoader(dataset = test_data, collate_fn = my_collate, shuffle = shuffle, batch_size = batch_size)

    return (train_loader, valid_loader, test_loader)

def train(epochs=10, learning_rate=0.01, batch_size=1, input_size=5200, hidden_size=2000, num_layers=5, shuffle=True):

    train_loader, valid_loader, test_loader = load_data(1, True)

    model = DynamicMLP(input_size, output_size, num_layers=num_layers, hidden_size=hidden_size)
    if GPU:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        model.train()
        start_time = time.time()
        total_loss = 0.0
        for i, (data, labels, hmm_seq) in enumerate(train_loader):
            data = Variable(torch.from_numpy(data), requires_grad=True).float()
            labels = Variable(torch.from_numpy(labels)).long()

            if GPU:
                data.cuda()
                labels.cuda()

            ouput = model(data, hmm_seq)

            loss = criterion(ouput, labels)
            loss.backward()

            optimizer.step()

            if (i+1) % ACCUM_GRAD == 0:
                optimizer.zero_grad()

            total_loss += loss.data[0]
            elapsed = time.time() - start_time
            s = 'Valid Epoch: {} [{}]\tLoss: {:.6f}\tTime: {:5.2f} '.format( epoch+1, i, total_loss/(i+1), elapsed)
            print s

        model.eval()
        start_time = time.time()
        total_loss = 0.0
        for i, (data, labels, hmm_seq) in enumerate(valid_loader):
            data = Variable(torch.from_numpy(data), requires_grad=True).float()
            labels = Variable(torch.from_numpy(labels)).long()

            if GPU:
                data.cuda()
                labels.cuda()

            ouput = model(data, hmm_seq)
            loss = criterion(ouput, labels)

            total_loss += loss.data[0]
            elapsed = time.time() - start_time
            s = 'Valid Epoch: {} [{}]\tLoss: {:.6f}\tTime: {:5.2f} '.format( epoch+1, i, total_loss/(i+1), elapsed)
            print s

train()

# feat = Variable(torch.from_numpy(feat), requires_grad=True).float()
# label = Variable(torch.from_numpy(label_h)).long()
# print label
#
# # audio = feat[0]
#
# dd = DynamicMLP(40, 2)
# # output = dd(audio, [1]*130)
# # print(output.grad)
#
# criterion = torch.nn.CrossEntropyLoss()
#
# optimizer = torch.optim.SGD(dd.parameters(), lr=1e-4, momentum=0.9)
# dd.train()
# for i in range(feat.shape[0]):
#     audio = feat[i]
#
#     output_audio = dd(audio, [1]*130)
#
#     loss = criterion(output_audio, label[i])
#     loss.backward()
#
#     optimizer.step()
#
#     if (i+1) % 10 == 0:
#         optimizer.zero_grad()
#
#     print i, loss.data[0]



# print output.shape
# print label[0].shape
# loss = criterion(output, label[0])
# loss.backward()
# print(loss.grad)
# print loss.data[0]
# optimizer.step()
