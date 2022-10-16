Author: Samuel Hale
Creation Date: 10/16/22
Purpose: Given a matrix of input data and a matrix of target data train a Linear Regression model on how to predict some target data only given input data using gradient descent.

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
