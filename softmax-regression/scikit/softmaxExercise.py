## Softmax Exercise 

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  softmax exercise using the scikit-learn package.

import argparse
import sys

import numpy as np

from sklearn import linear_model

parser = argparse.ArgumentParser('Softmax Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../../data/mnist/',
                    help='Directory to put the input MNIST data.')
parser.add_argument('-d', '--debug',
                    action='store_true',
                    help='Used for gradient checking.')

FLAGS, unparsed = parser.parse_known_args()

##======================================================================
## STEP 0: Initialise constants and parameters
#
#  Here we define and initialise some constants which allow your code
#  to be used more generally on any arbitrary input. 
#  We also initialise some parameters used for tuning the model.

inputSize = 28 * 28 # Size of input vector (MNIST images are 28x28)
numClasses = 10     # Number of classes (MNIST images fall into 10 classes)

decay = 1e-4 # Weight decay parameter

##======================================================================
## STEP 1: Load data
#
#  In this section, we load the input and output data.
#  For softmax regression on MNIST pixels, 
#  the input data is the images, and 
#  the output data is the labels.
#

images = np.load(FLAGS.input_data_dir + 'train-images.npy').T
labels = np.load(FLAGS.input_data_dir + 'train-labels.npy')

##======================================================================
## STEP 2: Training
#
#  Training your softmax regression model using L-BFGS.

numExamples = images.shape[0]

## ---------- YOUR CODE HERE --------------------------------------
# Compute the C parameter of the scikit objective formulation such that
# the resulting scikit objective is equivalent with the scipy objective.
C = 1.0/decay

## ---------- YOUR CODE HERE --------------------------------------
# Train a 'LogisticRegression' model using the 'multinomial' option
# for multiclass classification, and the C parameter computed above.
# Specify training with the L-BFGS solver for 100 max iterations.
softmax = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=C, fit_intercept=True,
 intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100,
  multi_class='multinomial', verbose=0, warm_start=False, n_jobs=1)
softmax.fit(images,labels)

# Although we only use 100 iterations here to train a classifier for the 
# MNIST data set, in practice, training for more iterations is usually
# beneficial.

##======================================================================
## STEP 5: Testing
#
#  You should now test your model against the test images.
#  To do this, you will first need to write code that computes 
#  predictions of a softmax model on the input data.

images = np.load(FLAGS.input_data_dir + 'test-images.npy').T
labels = np.load(FLAGS.input_data_dir + 'test-labels.npy')


## ---------- YOUR CODE HERE --------------------------------------
# Use the trained softmax model to pedict the labels 'pred' of test 'images'.
pred = softmax.predict(images)

acc = np.mean(labels == pred)
print('Accuracy: %0.3f%%.' % (acc * 100))

# Accuracy is the proportion of correctly classified images
# After 100 iterations, the results for our implementation were:
#
# Accuracy: 92.6%
#
# If your values are too low (accuracy less than 0.91), you should check 
# your code for errors, and make sure you are training on the 
# entire data set of 60000 28x28 training images 
# (unless you modified the loading code, this should be the case)
