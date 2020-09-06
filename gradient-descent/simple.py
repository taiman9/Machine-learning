#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Simple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la


# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  mean = np.zeros(X.shape[1])
  std = np.ones(X.shape[1])

  ## Your code here.
  mean=np.mean(X,axis=0)
  std=np.std(X,axis=0)
  return mean, std


# Standardize the features of the examples in X by subtracting their mean and 
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):
  S = np.zeros(X.shape)

  ## Your code here.
  S=(X-mean)/std
  return S

# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  data=np.loadtxt(file_name)
  X = data[:,:-1]
  t = data[:, data.shape[1]-1]
  return X, t



def train_normal_eq(X, t):
#  YOUR CODE here:

  A=np.dot(X.T,X)
  b=np.dot(X.T,t)
  w=np.dot((la.inv(A)),b)

  return w


# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, t, eta, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])
    
  for i in range(epochs):
    grad=compute_gradient(X, t, w)
    w=w-eta*grad
      
    if np.mod(i,10) == 0:
      cost=compute_cost(X, t, w)
      costs=np.append(costs,cost)
      ep=np.append(ep,i)
  return w,ep,costs

# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
#  YOUR CODE here:
  return np.sqrt(np.sum((np.sum(w.T*X,axis=1)-t)**2)/X.shape[0])


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
#  YOUR CODE here:
  return np.sum((np.sum(w.T*X,axis=1)-t)**2)/(2*X.shape[0])


# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X, t, w):
#  YOUR CODE here:
  grad = np.zeros(w.shape)
  # grad = X.T.dot((w.T).dot(X) - t)/N
  grad = np.dot(X.T,(np.sum(w.T*X,axis=1)-t))
  grad = grad/(X.shape[0])
  return grad



def train_SGD(X, t, eta, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])
    
  for i in range(epochs):
    for r in range(len(t)):
        xi=X[r,:]
        ti=t[r]
        grad=np.dot(xi.T,(np.sum(w.T*xi)-ti))
        w=w-eta*grad/X.shape[0]
      
    if np.mod(i,10) == 0:
      cost=compute_cost(X, t, w)
      costs=np.append(costs,cost)
      ep=np.append(ep,i)
  return w,ep,costs


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

#  YOUR CODE here: 
#     Make sure you add the bias feature to each training and test example.
#     Standardize the features using the mean and std comptued over *training*.

mean,std=mean_std(Xtrain)
Xtrain=standardize(Xtrain, mean, std)
Xtest=standardize(Xtest, mean, std)

X=Xtrain
Xt=Xtest
Xtrain=np.column_stack((np.ones(Xtrain.shape[0]),Xtrain))
Xtest=np.column_stack((np.ones(Xtest.shape[0]),Xtest))

# Computing paramters for each training method for eta=0.1 and 200 epochs
eta=0.1
epochs=200

w,eph,costs=train(Xtrain,ttrain,eta,epochs)
wn=train_normal_eq(Xtrain,ttrain)
wsgd,ephsgd,costssgd=train_SGD(Xtrain,ttrain,eta,epochs)


# Print model parameters.
print('Params GD: ', w)
print('Params SGD: ', wsgd)
print('Params Normal eq: ', wn)

# Plotting epochs vs. cost for gradient descent methods
plt.xlabel(' epochs')
plt.ylabel('cost')
plt.yscale('log')
plt.plot( eph,costs , 'bo-', label= 'train_jw_gd')
plt.plot( ephsgd,costssgd , 'ro-', label= 'train_j_w_sgd')
plt.legend()
plt.savefig('gd_cost_simple.png')
plt.close()

# Plotting linear approximation for each training method
plt.xlabel('Floor sizes')
plt.ylabel('House prices')
plt.plot( X,ttrain , 'bo', label= 'Training data')
plt.plot( Xt,ttest , 'g^', label= 'Test data')
plt.plot( X,w[0]+w[1]*X , 'b', label= 'GD')
plt.plot( X,wsgd[0]+wsgd[1]*X , 'r', label= 'SGD')
plt.plot( X,wn[0]+wn[1]*X , 'g', label= 'Normal eq')
plt.legend()
plt.savefig('train-test-line.png')
plt.close()


# In[ ]:




