import numpy as np
import pickle
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import*
import shutil
import os
from data_loader import*

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



## input parameters number of epochs, batch_size, shuffle
def train_MLP(epochs = 5, learning_rate = 0.01, batch_size = 1, input_size = 5200, hidden_size = 2000, shuffle = True):

    ## loads all the dataloaders. parameters (batch_size, shuffle)
    (train_loader, valid_loader, test_loader) = load_data(batch_size, shuffle)
    print ("DATA LOADED")

    net = Net(input_size, hidden_size, 2)
    net.cuda()
    print ("MODEL INITIALIZED")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
    for epoch in range (epochs):

        for i, (data, labels) in enumerate (train_loader):
##            print ("DATA SHAPE", data.shape)
            data = data.reshape(data.shape[0], 5200)
##            print ("Batch", i)
##            print ("LABEL SHAPE", labels.shape)
    ##        print ("DATA SHAPE", data.shape)
    ##        print ("Label shape", label.shape)

            ## converts to torch variables
            data = Variable(torch.from_numpy(data)).cuda().float()
            data = data.float()
            labels = Variable(torch.from_numpy(labels)).cuda()

            ## optimizers
            optimizer.zero_grad()
            outputs = net(data).float()
            loss = criterion(outputs, labels).float()
            loss.backward()
            optimizer.step()

            if (i+1) % 4 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                       %(epoch+1, epochs, i+1, 9, loss.data[0]))
        
        
    # Test the Model
    correct = 0
    total = 0
    for i, (data, labels) in enumerate (test_loader):
        data = data.reshape(data.shape[0], 5200)
        data = Variable(torch.from_numpy(data)).cuda().float()
        
        labels = Variable(torch.from_numpy(labels)).cpu()
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        print ("Total", total)
        print ("CORRECT", correct)

    print('Accuracy of the network on the test: %d %%' % (100 * (1.0*correct / total)))
    

train_MLP()
