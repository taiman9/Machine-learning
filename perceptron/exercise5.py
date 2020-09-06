
from perceptron import perceptron_train, aperceptron_train, perceptron_test, read_data 
from perceptron import kperceptron_train, kperceptron_test, quadratic_kernel
import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la


##======================= Main program =======================##
parser = argparse.ArgumentParser('Exercise 5')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/perceptron',
                    help='Directory for the perceptron dataset.')
FLAGS, unparsed = parser.parse_known_args()

print('\033[1m' + '1. Perceptron Convergence:\n' + '\033[0m')

# Read the training data for Exercise 5.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/perceptron.txt")

print('Exercise 5(a): \n')

# Train the perceptron algorithm on the training data.

w, weights, err = perceptron_train(Xtrain, ttrain, 10)

# Print the weight vector at different epochs.

print("Weight vector at the end of each epoch:\n")
ep = []
for i in range(weights.shape[0]-1):
    print("Epoch %d:" % (i+1))
    print(weights[i+1])
    ep = np.append(ep,i+1)
    
# Plot of errors at different epochs.

plt.xlabel('Epoch')
plt.ylabel("Errors")
plt.plot(ep, err, 'bo-', label= 'Total errors during epoch')
plt.legend()
plt.savefig('perceptron_5a.png')
plt.close()

print('\nExercise 5(b): \n')

# Train the kernel perceptron algorithm on the training data using a quadratic kernel.

alpha, a, err1 = kperceptron_train(Xtrain, ttrain, 15, quadratic_kernel)

# Show results of running the kernel perceptron algortihm on the training data.

print('Final values of Dual parameters:')
print(alpha)
print('\nDual parameters at the end of each epoch:\n')
for i in range(a.shape[0]-1):
    print("Epoch %d:" % (i+1))
    print(a[i+1])

# Show number of errors errors during each epoch.

print('\nNumber of errors during each epoch:')
ep = []
for i in range(err1.shape[0]):
    print("Epoch %d: %d" % (i+1, int(err1[i])))
    ep = np.append(ep,i+1)

plt.xlabel('Epoch')
plt.ylabel("Errors")
plt.plot(ep, err1, 'bo-', label= 'Total errors during epoch')
plt.legend()
plt.savefig('kperceptron_5b.png')
plt.close()


# In[ ]:





# In[ ]:




