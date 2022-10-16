"""
Author: Samuel Hale
Creation Date: 10/16/22
Purpose: Given a matrix of input data and a matrix of target data train a Linear Regression model on how to predict some target
data only given input data using gradient descent.

---------------------------------------
NOTES ON THIS CODE
---------------------------------------
Prerequisites to run this code:
1) Need Python v3. v3.9 was used for this specific program.
2) Need PyTorch installed v1.12.1 was used for this specific program
3) Need NumPy installed v1.23.4 was used for this specific program

This is a general program meant to take any input data and target data that can be modeled with linear regression (i.e. w11*input1 + w12*input2 + w13*input3 + b1 = target1).
This is my first model I've created, so it is not designed to be able to read in from files and initialize numpy arrays yet.
I am not currently sure if I will implement this during the lifespan of this file. Things that need to be changed if you want to use a different set of input and target data are:
1) Input matrix *found on line 37*
2) Target matrix *found on line 55*
3) Batch amount (if you want batches larger or smaller than 5) *found on line 86*
4) Learning rate *found on line 99*
5) The argument for how many epochs you want to train for *found on line 139 5th argument passed in*
----------------------------------------
"""

import torch
import numpy as np

#import the package that allows all of the utility functions below
import torch.nn as nn

#import data set utility
from torch.utils.data import TensorDataset

#import data loader
from torch.utils.data import DataLoader

#import loss functions
import torch.nn.functional as F 

#need data we will just use the data provided at the beginning of the nn.linear examples
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#Grab the amount of columns from inputs and targets
"""This will help us make this program more general so you could plug in any data that can be modeled by a linear regression"""
num_in_var = inputs.size(1)
num_targ_var = targets.size(1)

#need to initialize a dataset
"""In order to initialize a data set you just need inputs and targets passed in as arguments"""
training_ds = TensorDataset(inputs, targets)

#need to initialize a data loader
"""In order to initalize a data loader you need a data set, whether you want shuffling"""
batch_row_amt = 5 #batch size can also be thought of as the amount of rows you want to select from a data set
training_dl = DataLoader(training_ds, batch_row_amt, shuffle=True)

#need to initialize a linear regression model
"""For a model you just need to initalize it to a variable and pass in a tuple (...,...) the first entry in the tuple will be the amount of input variables needed, the second will be the amount of target variables this tells it how to shape the weight and bias matrix"""
linr_model = nn.Linear(num_in_var, num_targ_var)

#need to initalize a loss function
"""We will be using an mse loss function you don't need to pass any arguments in on initialization you just need to initalize it to a descriptive variable"""
loss_calc = F.mse_loss

#need to initilize an optimizer
"""We will be using an SGD optimizer which stands for Stochastic Gradient Descent(The Stochastic tells it we will be working with shuffled batches of data) you need to give it the model parameters and the learning rate since it needs those to look at gradients and adjust them and descriptive variable name"""
lr = 1e-5
optimize = torch.optim.SGD(linr_model.parameters(), lr)

#create and call a utility function that takes ^ above initializations in as arguments and trains the model for specified amount of epochs
"""The parameters should be (data loader, model, loss, optimizer, learning rate, epochs)"""
"""Follow the training process
1) Predictions
2) Loss
3) Calculate gradients of w.r.t variables
4) Optimize
5) Clear Gradients
"""
def train_it(training_dl, linr_model, loss_calc, optimize, num_epochs):
  #train it num_epochs times
  for epoch in range(num_epochs):

    #print progress
    if ((epoch+1)%10 == 0):
      print('Epoch[{},{}], Cost {:.4f}'.format(epoch+1, num_epochs, cost.item()))

    #initialize a new input batch and target batch to train it with
    for inb, tarb in training_dl:

      #predictions
      predict = linr_model(inb)

      #calculate loss which is interchangeably called cost
      cost = loss_calc(predict, tarb)

      #calculate gradients of w.r.t variables which are the weights and biases
      cost.backward()

      #optimize our weights and biases
      optimize.step()

      #clear our gradients so that the next gradient calculation doesn't sum them
      optimize.zero_grad()

#Show before and after results of how accurate predictions are
print('Predictions:\n{}, \nTargets:\n{}'.format(linr_model(inputs), targets))
train_it(training_dl, linr_model, loss_calc, optimize, 100)
print('Predictions:\n{}, \nTargets:\n{}'.format(linr_model(inputs), targets))