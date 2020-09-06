#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Simple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la

# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  data=np.loadtxt(file_name)
  X = data[:,:-1]
  t = data[:, data.shape[1]-1]
  return X, t


# Implement simple linear regression to compute w = [w0, w1].
def train(X, t):
# YOUR CODE here:
  X=np.column_stack((np.ones(X.shape[0]),X))
  A=np.dot(X.T,X)
  b=np.dot(X.T,t)
  w=np.dot((la.inv(A)),b)

  return w


# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
#  YOUR CODE here:
  X=np.column_stack((np.ones(X.shape[0]),X))
  sse = np.sum((np.sum(w.T*X,axis=1)-t)**2)
  rmse=np.sqrt(sse/X.shape[0])
  return rmse

# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
#  YOUR CODE here:
  X=np.column_stack((np.ones(X.shape[0]),X))
  cost=np.sum((np.sum(w.T*X,axis=1)-t)**2)/(2*X.shape[0])
  return cost


##======================= Main program =======================##
parser = argparse.ArgumentParser('Simple Regression Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/simple',
                    help='Directory for the simple houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

# Train model on training examples.
w = train(Xtrain, ttrain)

# Print model parameters.
print('Params: ', w)

# Print cost and RMSE on training data.
print('Training RMSE: %0.2f.' % compute_rmse(Xtrain, ttrain, w))
print('Training cost: %0.2f.' % compute_cost(Xtrain, ttrain, w))

# Print cost and RMSE on test data.
print('Test RMSE: %0.2f.' % compute_rmse(Xtest, ttest, w))
print('Test cost: %0.2f.' % compute_cost(Xtest, ttest, w))

#  YOUR CODE here: plot the training and test examples with different symbols,
#                  plot the linear approximation on the same graph.

plt.xlabel('Floor sizes')
plt.ylabel('House prices')
plt.plot( Xtrain,ttrain , 'bo', label= 'Training data')
plt.plot( Xtest,ttest , 'g^', label= 'Test data')
plt.plot( Xtrain,w[0]+w[1]*Xtrain , 'b', label= 'linear approx')
plt.legend()


plt.savefig('train_test_line.png')


# In[ ]:




