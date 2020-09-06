# Input:
#    'data' is a 2D array, with one exampel per row.
#    'labels' is a 1D array of labels for the corresponding examples.
#     'epochs' is the maximum number of epochs.
# Output:
#     the weight vector w.
import numpy as np 
def perceptron_train(data, labels, epochs):
	mistake=0
	w=np.zeros(data.shape[1])
	w1=np.zeros(data.shape[1])
	w1=w
	for eph in range(epochs):
		cnt=0
		for xi,t  in zip(data,labels):
			if np.sign(np.dot(w.T,xi)) != t:
				w=w+t*xi
				cnt=cnt+1
				mistake=mistake+1
	return w



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
	mistake=0
	wbar1=wbar
	for eph in range(epochs):
		cnt=0
		for xi,t in zip(data,labels):
			if np.sign(np.dot(w.T,xi)) != t:
				w=w+t*xi
				cnt=cnt+1
				mistake=mistake+1
			wbar=wbar+w
			tau=tau+1	
		if cnt == 0:
			print('avg perceptron converge after %s epochs:'%(eph+1))
			break
	w= np.reshape( w, (w.shape[0],1 ))		
	return wbar/tau,mistake




# Input:
#    'w' is the weight vector.
#    'data' is a 2D array, with one exampel per row.
# Output:
#     a vector with the predicted labels for the examples in 'data'.
def perceptron_test(w, data):
	h=np.dot(data,w)
	return np.hstack(np.sign(h))

def quadratic_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p



# Input:
#    'data' is a 2D array, with one exampel per row.
#    'labels' is a 1D array of labels for the corresponding examples.
#     'epochs' is the maximum number of epochs.
#     'kernel' is the kernel function to be used.
# Output:
#     the parameter vector alpha.
def kperceptron_train(data, labels, epochs, kernel=quadratic_kernel):
    examples = data.shape[0]
    alpha = np.zeros(examples, dtype=np.float64)

    # Gram matrix
    K = np.zeros((examples, examples))
    for i in range(examples):
        for j in range(examples):
            K[i,j] = kernel(data[i], data[j])
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







# Input:
#    'alpha' is the parameter vector.
#    'data' is a 2D array, with one exampel per row.
#     'kernel' is the kernel function to be used.
# Output:
#     a vector with the predicted labels for the examples in 'data'.
def kperceptron_test(data, alpha,kernel=quadratic_kernel):
    examples = data.shape[0]
    pred = np.zeros(examples, dtype=np.float64)
    # Gram matrix

    K = np.zeros((examples, examples))
   
    for i in range(examples):
        for j in range(examples):
            K[i,j] = kernel(data[i],data[j])
   
    for i in range(examples):
        pred[i]=np.sign(np.sum(K[:,i] * alpha ))
    return pred


# Input: two examples x and y.
# Output: the quadratic kernel value computed as (1+xTy)^2.
def quadratic_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p




# Input:
#    'file_name' is the name of the file containin a set of examples in the sparse feature vector format.
# Output:
#    a tuple '(data, labels)' where the 'data' is a two dimensional array containing all feature vectors, one per row, in the same order as in the input file, and the 'labels' is a vector containing the corresponding labels.

def read_examples(file_name):
	with open(file_name,'r') as f:
		tfs=[]
		ids=[]
		labels=[]
		for line in f:
			line_splited = line.split()
			labels = np.append(labels,int(line_splited[0]))
			id= [ int(a.split(':')[0]) for a in line_splited[1:] ]
			tf= [ float(a.split(':')[1]) for a in line_splited[1:] ]
			ids.append(np.array(id))
			tfs.append(np.array(tf))
		ids=np.array(ids)
		tfs=np.array(tfs)
	data=np.zeros((ids.shape[0],int(np.max(np.hstack(ids)))))
	data=np.append(arr = np.ones((data.shape[0],1)).astype(int), values = data, axis = 1)
	for i in range(ids.shape[0]):
		k=0
		for j in ids[i]:
			data[i,j]=tfs[i][k]
			k=k+1
	return data,labels


# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  dat=np.loadtxt(file_name)
  X = dat[:,:-1]
  t = dat[:, dat.shape[1]-1]
  return X, t



