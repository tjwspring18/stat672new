import scipy as sp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

'''
Note: I wrote the following function to generate the desired feature matrix
containing all quadratic and interaction terms of X. However, after doing so I
uncovered a scikitlearn function that does the exact same thing and so just use
that

def GeneratePhiMatrix(X):

	# get dimensions of X
	nrow = X.shape[0]
	ncol = X.shape[1]

	# preallocate final array 
	# this is *much* more memory-efficient than alternate approaches
	mat = np.zeros((nrow, int(ncol*2 + sp.special.comb(ncol, 2))))

	# first "chunk" of final array is just X
	mat[0:nrow, 0:ncol] = X

	# second "chunk" of final array is quadratic form of X
	mat[0:nrow, ncol+1:(2*ncol+1)] = pow(X, 2)

	# third "chunk" of final array is interaction terms

	# create empty list 
	lst = list()

	# generate all interaction terms 
	# apppend to list
	for i in range(0, ncol):
		for j in range(i+1, ncol):
			col = X[:,i] * X[:,j]
			lst.append(col)
	
	# assign columns from list to preallocated array
	for i in range(0, len(lst)):
		col = lst[i]
		mat[0:nrow, ncol*2+i] = col

	return(mat)
'''

# function to create test set consisting of every i'th observation
def GenerateTest(a, i):

	# get number of rows in a
	nrow = a.shape[0]

	# select every i'th row as part of test
	train = a[range(i-1, nrow+1, i)]

	return(train)

# function to create train set consisting of every observation not divisible by i
def GenerateTrain(a, i):

	# get number of rows in a
	nrow = a.shape[0]

	# list of numbers \leq number of rows and not divisible by i
	lst = list(range(1, nrow+1))
	for item in lst:
		if item % i == 0:
			lst.remove(item)
	for item in lst:
		item = item - 1
	
	# index observations not divisible by i
	train = a[lst]

	return(train)

# function to center matrix
def Center(a):

	# compute column means
	col_means = np.mean(a, axis=0)

	# subtract col means from relevant cols
	a = a - col_means

	return(a)

# function to scale matrix
def Scale(a):

	# square matrix
	s = pow(a, 2)

	# compute column sums of squared matrix
	s_squared = np.sum(s, axis = 0)

	# take square root of these sums
	s_squared_root = np.sqrt(s_squared)

	# divide original matrix by these square roots
	a = a / s_squared_root

	return(a)


# generate lambda

# compute SVD of X_train

# calculate \hat{w}_{ridge}

# function to compute test error

'''
# read data
X = np.genfromtxt('hw3_X.csv', delimiter = ',')
Y = np.genfromtxt('hw3_Y.csv', delimiter = ',')

# generate feature matrix
Phi = GeneratePhiMatrix(X)

#alternate way of doing so
poly = PolynomialFeatures(2)
Phi2 = poly.fit_transform(X)
Phi2 = np.delete(Phi2, 0, 1)

print(np.sum(Phi))
print(np.sum(Phi2))

# segregate into training and test sets

# train
trainPhi = GenerateTrain(Phi, 4)
trainY = GenerateTrain(Y, 4)

#test
testPhi = GenerateTest(Phi, 4)
testY = GenerateTest(Y,4)

# center and scale training sets
'''
