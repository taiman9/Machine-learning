#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Polynomial Curve Fitting Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la
# import scaling



# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  mean = np.zeros(X.shape[1])
  std = np.ones(X.shape[1])

  ## Your code here.
  std=np.std(X,axis=0)
  mean=np.mean(X,axis=0)
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
  
# train data with normal equations without regularization
def train_normal_eq(X, M, t):
#  YOUR CODE here:
  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  mean,std=mean_std(X[:,1:])
  X[:,1:]=standardize(X[:,1:], mean, std)
  w =np.dot(np.dot(la.inv(np.dot(X.T,X)),X.T),t)
  cost=compute_cost(X, t, w)
  return w,cost
 
# train data with normal equations with regularization   
def train_normal_eq_with_reg(X, M,lamda, t):
#  YOUR CODE here:
  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  mean,std=mean_std(X[:,1:])
  X[:,1:]=standardize(X[:,1:], mean, std)
  I=np.eye(M+1)
  N=len(t)
  w =np.dot(np.dot(la.inv(lamda*I*N+ np.dot(X.T,X)),X.T),t)
  cost=compute_cost_reg(X, t, w,lamda)
  return w,cost
 
    
# Compute gradient without regularization
def compute_gradient(X, t, w):
#  YOUR CODE here:
  grad = np.zeros(w.shape)
  grad = np.dot(X.T,(np.sum(w.T*X,axis=1)-t))
  grad = grad/(X.shape[0])
  return grad


# Compute gradient with regularization  
def compute_gradient_reg(X, t, w,lamda):
#  YOUR CODE here:
  grad = np.zeros(w.shape)
  grad = np.dot(X.T,(np.sum(w.T*X,axis=1)-t)) 
  grad = grad/(X.shape[0])+ lamda*w
  return grad

# Compute cost without regularization 
def compute_cost(X, t, w):
  return np.sum((np.sum(w.T*X,axis=1)-t)**2)/(2*X.shape[0])

# Compute cost with regularization
def compute_cost_reg(X, t, w,lamda):
  return np.sum((np.sum(w.T*X,axis=1)-t)**2)/(2*X.shape[0]) + 0.5*lamda*np.dot(w,w) 


# Batch Gradient Descent without regularization
def train(X, M,t, eta, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]

  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  mean,std=mean_std(X[:,1:])
  X[:,1:]=standardize(X[:,1:], mean, std)
  w = np.zeros(X.shape[1])
    
  costl=np.exp(21.0)
  for i in range(epochs):
    grad=compute_gradient(X, t, w)
    w=w-eta*grad
      
    cost = compute_cost(X, t, w)
    costs=np.append(costs,cost)
    ep=np.append(ep,i)

    if (costl-cost ) < np.exp(-10.0):
      break
    costl = cost
  return w,ep,costs  
  

# Batch Gradient Descent with regularization
def train_reg(X, M,t, eta, lamda, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  mean,std=mean_std(X[:,1:])
  X[:,1:]=standardize(X[:,1:], mean, std)
  w = np.zeros(X.shape[1])
    
  costl=np.exp(21.0)
  for i in range(epochs):
    grad=compute_gradient_reg(X, t, w,lamda)
    w=w-eta*grad
      
    cost = compute_cost_reg(X, t, w,lamda)
    costs=np.append(costs,cost)
    ep=np.append(ep,i)
      
    if (costl-cost ) < np.exp(-10.0):
      break
    costl = cost      
  return w,ep,costs  
  


# Stochastic Gradient Descent without regularization
def train_SGD(X,M, t, eta, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  mean,std=mean_std(X[:,1:])
  X[:,1:]=standardize(X[:,1:], mean, std)
  w = np.zeros(X.shape[1])
    
  costl=np.exp(21.0)
  for i in range(epochs):
    for r in range(len(t)):
        xi=X[r,:]
        ti=t[r]
        grad=np.dot(xi.T,(np.sum(w.T*xi)-ti))
        w=w-eta*(grad/X.shape[0])
      
    cost=compute_cost(X, t, w)
    costs=np.append(costs,cost)
    ep=np.append(ep,i)
    if (costl-cost ) < np.exp(-10.0):
      break
    costl = cost      
  return w,ep,costs


# Stochastic Gradient Descent with regularization  
def train_SGD_reg(X,M, t, eta, lamda,epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  mean,std=mean_std(X[:,1:])
  X[:,1:]=standardize(X[:,1:], mean, std)
  w = np.zeros(X.shape[1])
    
  costl=np.exp(21.0)
  for i in range(epochs):
    for r in range(len(t)):
        xi=X[r,:]
        ti=t[r]
        grad=np.dot(xi.T,(np.sum(w.T*xi)-ti)) + lamda*w
        w=w-eta*grad/X.shape[0]
      
    cost=compute_cost_reg(X, t, w,lamda)
    costs=np.append(costs,cost)
    ep=np.append(ep,i)
    if (costl-cost ) < np.exp(-10.0):
      break
    costl = cost      
  return w,ep,costs  
  
  
##======================= Main program =======================##
parser = argparse.ArgumentParser('Polyfit Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/polyfit',
                    help='Directory for the polyfit dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xdataset, tdataset = read_data(FLAGS.input_data_dir + "/dataset.txt")
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")
Xdevel, tdevel = read_data(FLAGS.input_data_dir + "/devel.txt")
#  YOUR CODE here: add the bias feature to each training and test example,
#                  create new design matrices X1train and X1test.



plt.xlabel(' x')
plt.ylabel('t(x)')
plt.plot( Xdataset,tdataset , 'bo-', label= 't(x)')
plt.legend()
plt.savefig('dataset.png')
plt.close()



plt.xlabel(' x')
plt.ylabel('t(x)')
plt.plot(Xtrain,ttrain, 'bo-', label= 'train data set')
plt.legend()
plt.savefig('traindataset.png')
plt.close()



plt.xlabel(' x')
plt.ylabel('t(x)')
plt.plot(Xtest,ttest, 'bo-', label= 'test data set')
plt.legend()
plt.savefig('testdataset.png')
plt.close()



plt.xlabel(' x')
plt.ylabel('t(x)')
plt.plot(Xdevel,tdevel, 'bo-', label= 'devel data set')
plt.legend()
plt.savefig('develdataset.png')
plt.close()


# M=5 gave the lowest RMSE in the first homework

M=5
epochs=1000

# Tuning the learning rate with the training data.

etas=[0.0001,0.001,0.01,0.1]
color=['r','b','g','k']
i=0

for eta in etas:
  w,eph,costs = train(Xtrain, M,ttrain, eta, epochs)
  plt.xlabel('epochs')
  plt.ylabel('J(w)')
  plt.plot(eph, costs, color[i], label= eta)
  plt.legend()
  i=i+1
plt.savefig('Epoch-vs-Jw-without-reg.png')
plt.close()

etas=[1.0,10.0]

i=0
for eta in etas:
  w,eph,costs = train(Xtrain, M,ttrain, eta, epochs)
  plt.xlabel('epochs')
  plt.ylabel('J(w)')
  plt.plot( eph,costs , color[i], label= eta)
  plt.legend()
  i=i+1
plt.savefig('Epoch-vs-Jw-without-reg2.png')
plt.close()

# Selecting eta=0.1 to plot epoch vs. cost without regularization

eta=0.1
epochs=2000
w,eph,costs = train(Xtrain,M, ttrain, eta, epochs)
wsgd,ephsgd,costssgd = train_SGD(Xtrain,M, ttrain, eta, epochs)
i=0
plt.xlabel('epochs')
plt.ylabel('J(w)')
plt.yscale('log')
plt.xscale('log')
plt.plot( eph,costs , color[i], label= 'GD')
plt.plot( ephsgd,costssgd , color[i+1], label= 'SGD')
plt.legend()
plt.savefig('Epoch-vs-J-without-reg-eta-01.png')
plt.close()

# Computing parameters for normal equations on training data
M=5
wn,costn=train_normal_eq(Xtrain, M, ttrain)

# Printing parameter values for each training method used
print('Params GD: ', w)
print('Params SGD: ', wsgd)
print('Params Normal eq: ', wn)


# Fixing M=9 and lamda tuned from first homework, repeating the experiments above with regularization

M=9

lamda=np.exp(-10)


epochs=1000
etas=[0.0001,0.001,0.01,0.1]
color=['r','b','g','k','m','c']
i=0

for eta in etas:
  w,eph,costs = train_reg(Xtrain, M,ttrain, eta, lamda,epochs)
  plt.xlabel('epochs')
  plt.ylabel('J(w)')
  plt.plot( eph,costs , color[i], label= eta)
  plt.legend()
  i=i+1
plt.savefig('Epoch-vs-Jw-with-reg.png')
plt.close()

etas=[1.0,10.0]

i=0
for eta in etas:
  w,eph,costs = train_reg(Xtrain, M,ttrain, eta, lamda,epochs)
  plt.xlabel('epochs')
  plt.ylabel('J(w)')
  plt.plot( eph,costs , color[i], label= eta)
  plt.legend()
  i=i+1
plt.savefig('Epoch-vs-Jw-with-reg2.png')
plt.close()


eta=0.1
epochs=2000
w,eph,costs = train_reg(Xtrain, M,ttrain, eta, lamda,epochs)
wsgd,ephsgd,costssgd = train_SGD_reg(Xtrain,M, ttrain, eta,lamda, epochs)
i=0
plt.xlabel('epochs')
plt.ylabel('J(w)')
plt.yscale('log')
plt.xscale('log')
plt.plot( eph,costs , color[i], label= 'GD')
plt.plot( ephsgd,costssgd , color[i+1], label= 'SGD')
plt.legend()
plt.savefig('Epoch-vs-J-with-reg-eta-01.png')
plt.close()

wn,costn=train_normal_eq_with_reg(Xtrain, M,lamda, ttrain)

print('Params GD: ', w)
print('Params SGD: ', wsgd)
print('Params Normal eq: ', wn)







# In[ ]:





# In[ ]:




