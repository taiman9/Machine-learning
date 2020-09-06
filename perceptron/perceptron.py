# Input:
#    'data' is a 2D array, with one exampel per row.
#    'labels' is a 1D array of labels for the corresponding examples.
#     'epochs' is the maximum number of epochs.
# Output:
#     the weight vector w.
import numpy as np 
def perceptron_train(data, labels, epochs):
	w=np.zeros(data.shape[1])
	weights=w
	error=[]
	for eph in range(epochs):
		cnt=0
		for xi,t  in zip(data,labels):
			if np.sign(np.dot(w.T,xi)) != t:
				w=w+t*xi
				cnt=cnt+1
		error = np.append(error,cnt)
		weights = np.vstack((weights,w))
		if cnt == 0:
			print('Converged after %s epochs.'%eph)
			break
	w= np.reshape( w, (w.shape[0],1 ))
	return w, weights, error



# Input:
#    'data' is a 2D array, with one exampel per row.
#    'labels' is a 1D array of labels for the corresponding examples.
#     'epochs' is the maximum number of epochs.
# Output:
#     the weight vector w.
def aperceptron_train(data, labels, epochs):
	w=np.zeros(data.shape[1])
	wbar=np.zeros(data.shape[1])
	tau=1
	error=[]
	for eph in range(epochs):
		cnt=0
		for xi,t in zip(data,labels):
			if np.sign(np.dot(w.T,xi)) != t:
				w=w+t*xi
				cnt=cnt+1
			wbar=wbar+w
			tau=tau+1	
		error = np.append(error,cnt)
		if cnt == 0:
			print('Converged after %s epochs.'%eph)
			break
	return wbar/tau, error




# Input:
#    'w' is the weight vector.
#    'data' is a 2D array, with one exampel per row.
# Output:
#     a vector with the predicted labels for the examples in 'data'.
def perceptron_test(w, data):
	h=np.dot(data,w)
	return np.hstack(np.sign(h))



# Input:
#    'data' is a 2D array, with one exampel per row.
#    'labels' is a 1D array of labels for the corresponding examples.
#     'epochs' is the maximum number of epochs.
#     'kernel' is the kernel function to be used.
# Output:
#     the parameter vector alpha.
def kperceptron_train(data, labels, epochs, kernel):
    n_samples, n_features = data.shape
    alpha = np.zeros(n_samples)
    a = np.zeros(n_samples)
    err = []
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(data[i], data[j])
    for eph in range(epochs):
        cnt = 0
        for i in range(n_samples):
            if np.sign(np.sum(K[:,i]*alpha*labels)) != labels[i]:
                alpha[i] = alpha[i]+1
                cnt=cnt+1
        a = np.vstack((a,alpha))
        err = np.append(err,cnt)
        if cnt == 0:
            print('Converged after %s epochs.'%eph)
            break
    return alpha, a, err




# Input:
#    'alpha' is the parameter vector.
#    'data' is a 2D array, with one exampel per row.
#     'kernel' is the kernel function to be used.
# Output:
#     a vector with the predicted labels for the examples in 'data'.
def kperceptron_test(alpha, data, kernel):
    n_samples = data.shape[0]
    K = np.zeros((n_samples, n_samples))
    h = []
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(data[i], data[j])
    for n in range(n_samples):
        fn = np.sign(np.sum(K[:,n]*alpha))
        h = np.append(h,fn)
    return h


# Input: two examples x and y.
# Output: the quadratic kernel value computed as (1+xTy)^2.
def quadratic_kernel(x, y):
    return (1 + np.dot(x.T, y))**2


# Input:
#    'file_name' is the name of the file containin a set of examples in the sparse feature vector format.
# Output:
#    a tuple '(data, labels)' where the 'data' is a two dimensional array containing all feature vectors, one per row, in the same order as in the input file, and the 'labels' is a vector containing the corresponding labels.
def read_examples(file_name):
	with open(file_name,'r') as f:
		labels=[]
		indexs=[]
		for line in f:
			elements = line.split()
			labels = np.append(labels,int(elements[0]))
			inds= [ int(a.split(':')[0]) for a in elements[1:] ]
			indexs.append(np.array(inds))
		indexs=np.array(indexs)
	data=np.zeros((indexs.shape[0],int(np.max(np.hstack(indexs)))))
	data=np.append(arr = np.ones((data.shape[0],1)).astype(int), values = data, axis = 1)
	for i in range(indexs.shape[0]):
		for j in indexs[i]:
			data[i,j]=1

	return data,labels


# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  dat=np.loadtxt(file_name)
  X = dat[:,:-1]
  t = dat[:, dat.shape[1]-1]
  return X, t



