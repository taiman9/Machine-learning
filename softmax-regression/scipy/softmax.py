import numpy as np
from scipy.sparse import coo_matrix

def softmaxCost(theta, numClasses, inputSize, decay, data, labels):
  """Computes and returns the (cost, gradient)

  numClasses - the number of classes 
  inputSize - the size N of the input vector
  lambda - weight decay parameter
  data - the N x M input matrix, where each row data[i, :] corresponds to
         a single sample
  labels - an M x 1 matrix containing the labels corresponding for the input data
  """

  # Unroll the parameters from theta
  theta = np.reshape(theta, (numClasses, inputSize))

  numCases = data.shape[1]

  groundTruth = coo_matrix((np.ones(numCases, dtype = np.uint8),
                            (labels, np.arange(numCases)))).toarray()
  cost = 0;
  thetagrad = np.zeros((numClasses, inputSize))
  
  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Compute the cost and gradient for softmax regression.
  #                You need to compute thetagrad and cost.
  #                The groundTruth matrix might come in handy.

  compute_matrix=theta.dot(data)
  overflow=np.max(compute_matrix)
  compute_matrix_scaled=compute_matrix-overflow
  compute_matrix_exp=np.exp(compute_matrix_scaled)
  compute_matrix_p= compute_matrix_exp/np.sum(compute_matrix_exp,axis=0)
  cost  = np.multiply(groundTruth, np.log(compute_matrix_p))
  cost = -(np.sum(cost) / data.shape[1])
  theta_squared = np.multiply(theta, theta)
  regu = 0.5 *decay  * np.sum(theta_squared)
  cost=cost+regu
  
 
  thetagrad = -np.dot(groundTruth - compute_matrix_p, data.T)
  thetagrad = thetagrad / data.shape[1] + decay * theta
  
  # ------------------------------------------------------------------
  # Unroll the gradient matrices into a vector for the optimization function.
  grad = thetagrad.ravel()

  return cost, grad


def softmaxPredict(theta, data):
  """Computes and returns the softmax predictions in the input data.

  theta - model parameters trained using fmin_l_bfgs_bin softmaxExercise.py,
          a numClasses x inputSize matrix.
  data - the M x N input matrix, where each row data[i,:] corresponds to
         a single sample.
  """

  #  Your code should produce the prediction matrix pred,
  #  where pred(i) is argmax_c P(c | x(i)).
 
  ## ---------- YOUR CODE HERE --------------------------------------
  #  Instructions: Compute pred using theta assuming that the labels start 
  #                from 0.
  pred=np.zeros(data.shape[1])
  compute_matrix=theta.dot(data)
  #overflow=np.max(compute_matrix)
  #compute_matrix=compute_matrix-overflow
  #compute_matrix_exp=np.exp(compute_matrix)
  #compute_matrix_p= compute_matrix_exp/np.sum(compute_matrix_exp,axis=0)
  #pred=np.argmax(compute_matrix_p, axis = 0)
  pred=np.argmax(compute_matrix, axis = 0)

  # ---------------------------------------------------------------------

  return pred
