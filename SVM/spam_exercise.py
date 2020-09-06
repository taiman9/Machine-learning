#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# Function to read data and labels from spam text files.
def read_data_txt(file_name):
	X=[]
	t=[]
	for line in open(file_name, 'r'):
		lineslt=line.split()
		X.append(lineslt[1:])
		t = np.append(t, int(lineslt[0]))
	return X, t




##======================= Main program =======================##
parser = argparse.ArgumentParser('Exercise spam')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/spam',
                    help='Directory for the spam dataset.')
FLAGS, unparsed = parser.parse_known_args()

print('\033[1m' + '2. Spam vs. Non-Spam:\n' + '\033[0m')

X=[]
t=[]
unique_words=[]

# Read spam training data.

X,t=read_data_txt(FLAGS.input_data_dir + '/spam_train.txt')

# Create a vocabulary as instructed in 2(a).

vocab=dict()
vocab_30=dict()

for line1 in X:
	unique_words=set(line1)
	for word in unique_words:
		if word in vocab:
			vocab[word] += 1
		else:
			vocab[word]= 1 

vocab_30 = {key: value for key, value in vocab.items() if value>=30}
vocab_30_sorted = sorted(vocab_30.keys())
vocab_final = {key: value for key,value in enumerate(vocab_30_sorted,1)}


# Create a vocabulary file spam_vocab.txt (Commented out as already created).

'''
f = open('spam_vocab.txt','w')
for key, value in vocab_final.items():
	f.write('{} {}\n'.format(key,value))
f.close()

'''

# Create files spam_train_svm.txt and spam_test_svm.txt as instructed in 2(b) (Commented out as already created).

'''

with open(FLAGS.input_data_dir + '/spam_train.txt','r') as f:
	Sparse_features = []
	for line in f:
		elements = line.split()
		l1 = []
		label = str(elements[0])
		words = set(elements[1:])
		for word in words:
			for key,value in vocab_final.items():
				if value == word:
					l1.append(key)
		l1 = list(sorted(l1))
		l1 = [str(s) + ':1' for s in l1]
		l1.insert(0,label)
		Sparse_features.append(l1)
	with open('spam_train_svm.txt', 'w') as f:
		for _list in Sparse_features:
			f.write(' '.join(_list) + '\n')

'''

# Read training and test data using read_examples function from the sparse feature vector representations.

data,labels=read_examples(FLAGS.input_data_dir + '/spam_train_svm.txt')
data1,labels1=read_examples(FLAGS.input_data_dir + '/spam_test_svm.txt')

print("SVM algorithm: \n")
svm = svm.SVC(C= 5,kernel='linear')
svm.fit(data, labels)

svm_test = svm.predict(data1)
accuracy = np.mean(labels1 == svm_test)
print("Accuracy of SVM with linear kernel: %0.2f%%." %(accuracy * 100))


print("\nPerceptron algorithm: \n")

# Train the perceptron algorithm on the training data.

w, _, error = perceptron_train(data, labels, 50)

# Save the returned parameter vector in spam_model_p.txt.

p = np.savetxt('spam_model_p.txt', w) 

# Report number of mistakes during each epoch and total number of mistakes during training.

print("\nNumber of mistakes during each epoch:")

eph = 1
for i in error:
    print("Epoch %d: %d" % (eph, i))
    eph += 1 

print("\nTotal number of mistakes: %d" % np.sum(error))

plt.xlabel('Epochs')
plt.ylabel("Errors")
plt.plot(error, 'bo-', label= 'Total errors during epoch')
plt.legend()
plt.savefig('perceptron_train.png')
plt.close()

# Test the perceptron algorithm  on the test data by reading the parameter vector from spam_model_p.txt.

with open('spam_model_p.txt', 'r') as f:
    w1 = f.readlines()

w1 = np.asarray(w1, dtype=np.float64)
w1 = np.reshape(w1, (-1, 1))

pred1 = perceptron_test(w1, data1)

# Report test accuracy.

acc1 = np.mean(labels1 == pred1)
print('\nAccuracy on test data: %0.2f%%.' % (acc1 * 100))

# Carry out same procedure for Average perceptron algorithm as done for the Vanilla perceptron algorithm.

print("\nAverage Perceptron algorithm: \n")

aw, aerror = aperceptron_train(data, labels, 50)

ap = np.savetxt('spam_model_ap.txt', aw) 
    
print("\nNumber of mistakes during each epoch:")

eph = 1
for i in aerror:
    print("Epoch %d: %d" % (eph, i))
    eph += 1 

print("\nTotal number of mistakes: %d" % np.sum(aerror))

plt.xlabel('Epochs')
plt.ylabel("Errors")
plt.plot(aerror, 'bo-', label= 'Total errors during epoch')
plt.legend()
plt.savefig('aperceptron_train.png')
plt.close()

with open('spam_model_ap.txt', 'r') as f:
    w2 = f.readlines()

w2 = np.asarray(w2, dtype=np.float64)
w2 = np.reshape(w2, (-1, 1))

pred2 = perceptron_test(w2, data1)

acc2 = np.mean(labels1 == pred2)
print('\nAccuracy on test data: %0.2f%%.' % (acc2 * 100))




# In[ ]:




