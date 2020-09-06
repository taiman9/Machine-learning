import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la
import timeit
from sklearn import svm
from perceptron import perceptron_train
from perceptron import aperceptron_train
from perceptron import perceptron_test
from perceptron import read_examples
from sklearn.metrics import confusion_matrix
def read_data(file_name):
    dat=np.loadtxt(file_name,delimiter=',')
    X = dat[:,:-1]
    t = dat[:, dat.shape[1]-1]
    return X, t


def scalling(X):
    S = np.zeros(X.shape)
    range1=X.shape[1]
    max1 = np.zeros(X.shape[1])
    min1 = np.ones(X.shape[1])
    max1=np.max(X,axis=0)
    min1=np.min(X,axis=0)
   
    for i in range(range1):
        if max1[i]==min1[i]:
            S[:,i]=X[:,i]
            if max1[i] != 0:
                S[:,i]=S[:,i]/max1[i]
        else :
            S[:,i]=(X[:,i]-min1[i])/(max1[i]-min1[i])
    return S

def kernel(x, y, p):
    return (1 + np.dot(x, y)) ** p


def kperceptron_train(data, labels, epochs, p):
    examples = data.shape[0]
    alpha = np.zeros(examples, dtype=np.float64)

    # Gram matrix
    K = np.zeros((examples, examples))
    for i in range(examples):
        for j in range(examples):
            K[i,j] = (1+ np.dot(data[i],data[j]))**p
    for eph in range(epochs):
        cnt=0
        for i in range(examples):
            if np.sign(np.sum(K[:,i] * alpha )) != labels[i]:
                alpha[i] += labels[i]
                cnt=cnt+1
        if cnt == 0:
            print('kernel perceptron converge after %s epochs:'%(eph+1))
            break
    return alpha





def kperceptron_test(data, data_t, alpha,p):
    examples = data.shape[0]
    examples_t=data_t.shape[0]
    pred = np.zeros(examples_t, dtype=np.float64)

    K = np.zeros((examples, examples_t))
   
    for i in range(examples):
        for j in range(examples_t):
            K[i,j] = (1+ np.dot(data[i],data_t[j]))**p
   
    for i in range(examples_t):
        pred[i]=np.sum(K[:,i] * alpha )
    return pred


def kperceptron_train(data, labels, epochs, p):
    examples = data.shape[0]
    alpha = np.zeros(examples, dtype=np.float64)

    # Gram matrix
    K = np.zeros((examples, examples))
    for i in range(examples):
        for j in range(examples):
            K[i,j] = (1+ np.dot(data[i],data[j]))**p
    for eph in range(epochs):
        cnt=0
        for i in range(examples):
            if np.sign(np.sum(K[:,i] * alpha )) != labels[i]:
                alpha[i] += labels[i]
                cnt=cnt+1
        if cnt == 0:
            print('kernel perceptron converge after %s epochs:'%(eph+1))
            break
    return alpha

def kgperceptron_train(data, labels, epochs, sig):
    examples = data.shape[0]
    alpha = np.zeros(examples, dtype=np.float64)

    # Gram matrix
    K = np.zeros((examples, examples))
    for i in range(examples):
        for j in range(examples):
            K[i,j] =  np.exp(-np.dot( data[i]-data[j], data[i]-data[j])/sig**2.0)
    for eph in range(epochs):
        cnt=0
        for i in range(examples):
            if np.sign(np.sum(K[:,i] * alpha )) != labels[i]:
                alpha[i] += labels[i]
                cnt=cnt+1
        if cnt == 0:
           # print('kernel perceptron converge after %s epochs:'%(eph+1))
            break
    return alpha




def kperceptron_test(data, data_t, alpha,p):
    examples = data.shape[0]
    examples_t=data_t.shape[0]
    pred = np.zeros(examples_t, dtype=np.float64)

    K = np.zeros((examples, examples_t))
   
    for i in range(examples):
        for j in range(examples_t):
            K[i,j] = (1+ np.dot(data[i],data_t[j]))**p
   
    for i in range(examples_t):
        pred[i]=np.sum(K[:,i] * alpha )
    return pred


def kgperceptron_test(data, data_t, alpha,sig):
    examples = data.shape[0]
    examples_t=data_t.shape[0]
    pred = np.zeros(examples_t, dtype=np.float64)

    K = np.zeros((examples, examples_t))
   
    for i in range(examples):
        for j in range(examples_t):
            K[i,j] =np.exp(-np.dot( data[i]-data_t[j], data[i]-data_t[j])/sig**2.0)
   
    for i in range(examples_t):
        pred[i]=np.sum(K[:,i] * alpha )
    return pred





##======================= Main program =======================##
parser = argparse.ArgumentParser('Exercise spam')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/digit',
                    help='Directory for the perceptron dataset.')
FLAGS, unparsed = parser.parse_known_args()



X,t=read_data(FLAGS.input_data_dir + '/optdigits.tra')
X_sc=scalling(X)


X_t,t_t=read_data(FLAGS.input_data_dir + '/optdigits.tes')
X_t_sc=scalling(X_t)


t10=np.zeros((len(t),10))
for i in range(10):
    for j in range(len(t)):
        if t[j]==i:
            t10[j,i]=1
        else:
            t10[j,i]=-1

accrucys=[]
for j in range(1,21):
    pred_ehp=[]
    for i in range(10):
        w= perceptron_train(X_sc[1000:,:], t10[1000:,i], j)
        w=w/np.linalg.norm(w)
        pred=X_sc[:1000,:].dot(w)
        pred_ehp.append(pred)
    labels=np.argmax(pred_ehp,axis=0)
    ac=np.mean(labels==t[:1000])*100
    print(j,ac)
    accrucys.append(ac)    
print('max accrucy in epoch:',np.argmax(accrucys)+1) 
set_eph=8
set_eph=np.argmax(accrucys)+1

st=timeit.default_timer()
pred_ehp=[]
for i in range(10):
    w= perceptron_train(X_sc[1000:,:], t10[1000:,i], set_eph)
    w=w/np.linalg.norm(w)
    pred=X_t_sc[:,:].dot(w)
    pred_ehp.append(pred)
labels=np.argmax(pred_ehp,axis=0)
ac=np.mean(labels==t_t)*100
print('accrucy for linear perceptron' ,ac)
sp=   timeit.default_timer() 

print('time takes for linear perceptron',sp-st)
print('confusion_matrix')

print(confusion_matrix(t_t, labels))

set_eph=8

accrucys=[]
for j in range(2,7):
    pred_ehp=[]
    for i in range(10):
        w= kperceptron_train(X_sc[1000:,:], t10[1000:,i], 8, j)
        pred=kperceptron_test(X_sc[1000:,:],X_sc[:1000,:],w,j)
        pred_ehp.append(pred)
    labels=np.argmax(pred_ehp,axis=0)
    ac=np.mean(labels==t[:1000])*100
    print(j,ac)
    accrucys.append(ac)    
print('max accrucy in d poly:',np.argmax(accrucys)+2) 


set_p=np.argmax(accrucys)+2

set_eph=8
set_p=6   
sv1=0
st=timeit.default_timer()
pred_ehp=[]
for i in range(10):
    w= kperceptron_train(X_sc[1000:,:], t10[1000:,i], set_eph, set_p)
    sv = w > 1e-5
    sv1=sv1+len(sv) 
    pred=kperceptron_test(X_sc[1000:,:],X_t_sc,w,set_p)
    pred_ehp.append(pred)
labels=np.argmax(pred_ehp,axis=0)
ac=np.mean(labels==t_t)*100
print('accrucy for poly kernel perceptron',ac)

sp=   timeit.default_timer() 
print('number of suppport vector',sv1)
print('time takes for poly kenael perceptron',sp-st)

print('confusion_matrix for poly ')

print(confusion_matrix(t_t, labels))



sigma=[0.1,0.5,2.0,5.0,10.0]
accrucys=[]
for sig in sigma:
    pred_ehp=[]
    for i in range(10):
        w= kgperceptron_train(X_sc[1000:,:], t10[1000:,i], set_eph, sig)
        pred=kgperceptron_test(X_sc[1000:,:],X_sc[:1000,:],w,sig)
        pred_ehp.append(pred)
    labels=np.argmax(pred_ehp,axis=0)
    ac=np.mean(labels==t[:1000])*100
    print(sig,ac)
    accrucys.append(ac)    
print('max accrucy in sig:', sigma(np.argmax(accrucys))) 
set_p=np.argmax(accrucys)

sig=2.0
sv1=0
st=timeit.default_timer()
pred_ehp=[]
for i in range(10):
    w= kgperceptron_train(X_sc[1000:,:], t10[1000:,i], 8, sig)
    pred=kgperceptron_test(X_sc[1000:,:],X_t_sc,w,sig)
    sv = w > 1e-5
    sv1=sv1+len(sv) 
    pred_ehp.append(pred)
labels=np.argmax(pred_ehp,axis=0)
ac=np.mean(labels==t_t)*100
print('accrucy for gaussian kernel perceptron',ac)
print('number of suppport vector',sv1)
sp=   timeit.default_timer() 

print('time takes for gaussian kenael perceptron',sp-st)
print('confusion_matrix for gaussian kenel ')
print(confusion_matrix(t_t, labels))





st=timeit.default_timer()
svm_tra = svm.SVC(kernel='linear',  C = 1.0,decision_function_shape='ovo') 
svm_tra.fit(X[1000:,:], t[1000:])
pred = svm_tra.predict(X_t)

ac=np.mean(pred==t_t)*100
print('accrucy for linear svm',ac)

sp=   timeit.default_timer() 
print('time takes for linear svm',sp-st)

print('confusion_matrix for linear svm ')

print(confusion_matrix(t_t, pred))





accrucys=[]
for j in range(2,7):
    svm_tra = svm.SVC(kernel = 'poly', degree = j, C = 1.0,decision_function_shape='ovo') 
    svm_tra.fit(X[1000:,:], t[1000:])
    pred = svm_tra.predict(X[:1000,:])
    ac=np.mean(pred==t[:1000])*100
    print(j,ac)
    accrucys.append(ac)    
print('max accrucy in d poly:',np.argmax(accrucys)+2) 
set_p=5
set_p=np.argmax(accrucys)+2

 
st=timeit.default_timer()
svm_tra = svm.SVC(kernel = 'poly', degree = set_p, C = 1.0,decision_function_shape='ovo') 
svm_tra.fit(X[1000:,:], t[1000:])
pred = svm_tra.predict(X_t)

ac=np.mean(pred==t_t)*100
print('accrucy for poly kernel svm',ac)

sp=   timeit.default_timer() 
print('time takes for poly kenael svm',sp-st)

print('confusion_matrix for poly svm ')

print(confusion_matrix(t_t, pred))







sigma=[0.1,0.5,2.0,5.0,10.0]

accrucys=[]
for j in sigma:
    svm_tra = svm.SVC(kernel = 'rbf', gamma = 1./(2. * j**2), C = 1.0,decision_function_shape='ovo') 
    svm_tra.fit(X[1000:,:], t[1000:])
    pred = svm_tra.predict(X[:1000,:])
    ac=np.mean(pred==t[:1000])*100
    print(j,ac)
    accrucys.append(ac)    
print('max accrucy in sig gaussian:',np.argmax(accrucys)+2) 

sig=10.0 
st=timeit.default_timer()
svm_tra = svm.SVC(kernel = 'rbf', gamma = 1./(2. * sig**2), C = 1.0,decision_function_shape='ovo') 
svm_tra.fit(X[1000:,:], t[1000:])
pred = svm_tra.predict(X_t)

ac=np.mean(pred==t_t)*100
print('accrucy for gaussian svm',ac)

sp=   timeit.default_timer() 
print('time takes for gaussian svm',sp-st)

print('confusion_matrix for gaussian svm ')

print(confusion_matrix(t_t, pred))







