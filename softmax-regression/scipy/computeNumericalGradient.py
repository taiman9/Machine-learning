import numpy as np

def computeNumericalGradient(J, theta):
  """ Compute numgrad = computeNumericalGradient(J, theta)

  theta: a vector of parameters
  J: a function that outputs a real-number and the gradient.
  Calling y = J(theta)[0] will return the function value at theta. 
  """

  # Initialize numgrad with zeros
  numgrad = np.zeros(theta.size)

  #print(theta)
  ## ---------- YOUR CODE HERE --------------------------------------
  # Instructions: 
  # Implement numerical gradient checking, and return the result in numgrad.  
  # You should write code so that numgrad(i) is (the numerical approximation to) the 
  # partial derivative of J with respect to the i-th input argument, evaluated at theta.  
  # I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
  # respect to theta(i).
  #                
  # Hint: You will probably want to compute the elements of numgrad one at a time. 

  epsilon = 0.0001

  theta_p=np.array(theta,dtype=np.float)
  theta_n=np.array(theta,dtype=np.float)

  for i in range(len(numgrad)):
    
    theta_p[i] = theta[i] + epsilon
    theta_n[i] = theta[i] - epsilon
    
    numgrad[i]=( J(theta_p)[0] -J(theta_n)[0])/(2*epsilon)

    theta_p[i] = theta[i]     
    theta_n[i] = theta[i]

  ## ---------------------------------------------------------------

  return numgrad
