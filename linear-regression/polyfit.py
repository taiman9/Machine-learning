#!/usr/bin/env python
# coding: utf-8

# In[1]:


## polyfit Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la

# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  data=np.loadtxt(file_name)
  X = data[:,0]
  t = data[:, 1]

  return X, t


# Implement normal equations to compute w = [w0, w1, ..., w_M].
def train(X, M, t):
#  YOUR CODE here:
  X=X.T
  X=(np.vander(X, M+1,increasing=True))
  A=np.dot(X.T,X)
  b=np.dot(X.T,t)
  w=np.dot((la.inv(A)),b)
  return w


def train_reg(X, M,lamda, t):
#  YOUR CODE here:
  X=X.T
  X=(np.vander(X, M+1,increasing=True))
  N=len(t)
  I=np.eye(M+1)
  A=np.dot(X.T,X)
  b=np.dot(X.T,t)
  w=np.dot((la.inv(lamda*I*N + A)),b)
  return w




# Compute RMSE on dataset (X, t).
def compute_rmse(X,M ,t, w):
#  YOUR CODE here:\
  X=X.T
  X=(np.vander(X, M+1,increasing=True))
  rmse = np.sqrt(np.sum((np.sum(w.T*X,axis=1)-t)**2)/X.shape[0])
  return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, M, t, w):
#  YOUR CODE here:
  X=X.T
  X=(np.vander(X, M+1,increasing=True))
  cost = np.sum((np.sum(w.T*X,axis=1)-t)**2)/(2*X.shape[0])
  return cost


##======================= Main program =======================##
parser = argparse.ArgumentParser('Poly fit Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/polyfit',
                    help='Directory for the polyift dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xdataset, tdataset = read_data(FLAGS.input_data_dir + "/dataset.txt")
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")
Xdevel, tdevel = read_data(FLAGS.input_data_dir + "/devel.txt")
#  YOUR CODE here: add the bias feature to each training and test example,
#                  create new design matrices X1train and X1test.

X1train = Xtrain
X1test = Xtest

# Plot the data in dataset.txt.
plt.xlabel(' x')
plt.ylabel('t(x)')
plt.plot( Xdataset,tdataset , 'bo-', label= 't(x)')
plt.legend()
plt.savefig('dataset.png')
plt.close()


# Plot the data in train.txt.
plt.xlabel(' x')
plt.ylabel('t(x)')
plt.plot(Xtrain,ttrain, 'bo-', label= 'training data set')
plt.legend()
plt.savefig('traindataset.png')
plt.close()


# Plot the data in test.txt.
plt.xlabel(' x')
plt.ylabel('t(x)')
plt.plot(Xtest,ttest, 'bo-', label= 'test data set')
plt.legend()
plt.savefig('testdataset.png')
plt.close()


# Plot the data in devel.txt.
plt.xlabel(' x')
plt.ylabel('t(x)')
plt.plot(Xdevel,tdevel, 'bo-', label= 'devel data set')
plt.legend()
plt.savefig('develdataset.png')
plt.close()


M=9
train_rmse=[]
test_rmse=[]
order=[]

for i in range(M+1):
# Train model without relularization for M ∈ [0, 9] on training data.
  w = train(X1train,i, ttrain)
  order=np.append(i,order)

# Compute RMSE on training data.
  a=compute_rmse(X1train, i, ttrain, w)
  train_rmse=np.append(a, train_rmse)

# Compute RMSE on test data.
  b=compute_rmse(X1test,i, ttest, w)
  test_rmse=np.append(b,test_rmse)

# Plot training and test RMSE values for M ∈ [0, 9].
plt.xlabel('M')
plt.ylabel('RMSE')
plt.plot( order,train_rmse , 'bo-', label= 'train_rmse')
plt.plot( order,test_rmse , 'go-', label= 'test_rmse')
plt.legend()
plt.savefig('train-test-rmse-without-reg.png')
plt.close()




M=9

train_rmse=[]
devel_rmse=[]
model=[]

for i in range(0,51,5):
  # For ln λ ∈ [−50, 0] in steps of 5, train model with regularization on training data.
  lamda=np.exp(i-50*1.0)
  w = train_reg(X1train,M,lamda, ttrain)
  model=np.append(i-50,model)
  # Compute RMSE on training data.
  a=compute_rmse(X1train, M, ttrain, w)
  train_rmse=np.append(a, train_rmse)
  # Compute RMSE on validation data.
  b=compute_rmse(Xdevel,M, tdevel, w)
  devel_rmse=np.append(b,devel_rmse)
  
 # Plot training and validation RMSE values for ln λ ∈ [−50, 0] and M=9.
plt.xlabel(' ln lamda')
plt.ylabel('RMSE')
plt.plot( model,train_rmse , 'bo-', label= 'train_rmse')
plt.plot( model,devel_rmse , 'go-', label= 'devel_rmse')
plt.legend()
plt.savefig('train-devel-rmse-with-reg.png')
plt.close()


  # Train model without regularization and print RMSE for M=9.
w = train(X1train,M, ttrain)
b=compute_rmse(X1test,M, ttest, w)
print('Test without regularization for M =%2i, RMSE: %0.2f.' %( M,b))
  # Train model with regularization for ln λ giving lowest RMSE for M=9.
lamda=np.exp(-10)
w = train_reg(X1train,M,lamda, ttrain)
b=compute_rmse(X1test,M, ttest, w)
print('Test with regularization for M =%2i, RMSE: %0.2f.' %( M,b))










# In[ ]:




