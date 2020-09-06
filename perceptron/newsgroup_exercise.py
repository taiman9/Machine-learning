
import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la

from perceptron import perceptron_train
from perceptron import aperceptron_train
from perceptron import perceptron_test

from perceptron import read_examples

##======================= Main program =======================##
parser = argparse.ArgumentParser('Exercise newsgroups')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/newsgroups',
                    help='Directory for the newsgroups dataset.')
FLAGS, unparsed = parser.parse_known_args()

print('\033[1m' + '3. Atheism vs. Religion:\n' + '\033[0m')

# Read the training and test data.

trdata1, trlabels1 = read_examples(FLAGS.input_data_dir + "/newsgroups_train1.txt")
trdata2, trlabels2 = read_examples(FLAGS.input_data_dir + "/newsgroups_train2.txt")

tsdata1, tslabels1 = read_examples(FLAGS.input_data_dir + "/newsgroups_test1.txt")
tsdata2, tslabels2 = read_examples(FLAGS.input_data_dir + "/newsgroups_test2.txt")

# Train the perceptron algorithm on training datasets for 10,000 epochs.

w1, ws1, error1 = perceptron_train(trdata1, trlabels1, 10000)
w2, ws2, error2 = perceptron_train(trdata2, trlabels2, 10000)

# Save the returned parameter vectors from training in appropriate text files.

p1 = np.savetxt('newsgroups_model_p1.txt', w1) 

p2 = np.savetxt('newsgroups_model_p2.txt', w2) 

# Evaluate the perceptron algorithm on the corresponding test examples for each version by reading the parameter vectors
# from the corresponding text files.

with open('newsgroups_model_p1.txt', 'r') as f:
    wp1 = f.readlines()

wp1 = np.asarray(wp1, dtype=np.float64)
wp1 = np.reshape(wp1, (-1, 1))

with open('newsgroups_model_p2.txt', 'r') as f:
    wp2 = f.readlines()

wp2 = np.asarray(wp2, dtype=np.float64)
wp2 = np.reshape(wp2, (-1, 1))

pred1 = perceptron_test(wp1, tsdata1)
pred2 = perceptron_test(wp2, tsdata2)

# Report test accuracy.

acc1 = np.mean(tslabels1 == pred1)
print('Accuracy of perceptron on test dataset version 1: %0.3f%%.' % (acc1 * 100))

acc2 = np.mean(tslabels2 == pred2)
print('Accuracy of perceptron on test dataset version 2: %0.3f%%.' % (acc2 * 100))

# Carry out same procedure for the average perceptron algorithm as that for the perceptron algorithm.

aw1, aerror1 = aperceptron_train(trdata1, trlabels1, 10000)
aw2, aerror2 = aperceptron_train(trdata2, trlabels2, 10000)

ap1 = np.savetxt('newsgroups_model_ap1.txt', aw1) 
ap2 = np.savetxt('newsgroups_model_ap2.txt', aw2) 

with open('newsgroups_model_ap1.txt', 'r') as f:
    wap1 = f.readlines()

wap1 = np.asarray(wap1, dtype=np.float64)
wap1 = np.reshape(wap1, (-1, 1))

with open('newsgroups_model_ap2.txt', 'r') as f:
    wap2 = f.readlines()

wap2 = np.asarray(wap2, dtype=np.float64)
wap2 = np.reshape(wap2, (-1, 1))


apred1 = perceptron_test(wap1, tsdata1)
apred2 = perceptron_test(wap2, tsdata2)

acc1 = np.mean(tslabels1 == apred1)
print('Accuracy of average perceptron on test dataset version 1: %0.3f%%.' % (acc1 * 100))

acc2 = np.mean(tslabels2 == apred2)
print('Accuracy of average perceptron on test dataset version 2: %0.3f%%.' % (acc2 * 100))


# In[ ]:




