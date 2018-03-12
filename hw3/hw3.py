#TODO: email Slawski about scaling and centering - skl?
import scipy as sp
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Ridge

'''
Note: I wrote the following function to generate the desired feature matrix
containing all quadratic and interaction terms of X. However, after doing so I
uncovered a scikitlearn function that does the exact same thing and so just use
that
'''
'''
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

# function to center and scale matrix
def CenterAndScale(a):

	# compute column means
	col_means = np.mean(a, axis=0)

	# subtract col means to center matrix
	cen = a - col_means

	# compute L2 norms of columns of centered matrix
	cen_norm = np.sqrt(np.sum(pow(cen, 2), axis = 0))

	# divide centered matrix by column norms
	out = cen / cen_norm
	
	return(out)

# function to compute S_Lambda
def ComputeS_Lambda(S, Lambda):

	# number of coefficients 
	# same as number of singular values
	n = len(S)

	# make s_lambda matrix
	
	# initialize as n x n zero matrix
	S_Lambda = np.zeros((n, n))

	# fill in along diagonal
	vals = S / (pow(S, 2) + Lambda)
	np.fill_diagonal(S_Lambda, vals)

	return(S_Lambda)

# read data
X = np.genfromtxt('hw3_X.csv', delimiter = ',')
Y = np.genfromtxt('hw3_Y.csv', delimiter = ',')

# generate feature matrix Phi
poly = preprocessing.PolynomialFeatures(2)
Phi = poly.fit_transform(X)

# delete first column of Phi (which is all 1's)
Phi = np.delete(Phi, 0, 1)

# segregate into training and test sets

# train
trainPhi = GenerateTrain(Phi, 4)
trainY = GenerateTrain(Y, 4)

# test
testPhi = GenerateTest(Phi, 4)
testY = GenerateTest(Y, 4)

# center and scale training X
trainPhi = CenterAndScale(trainPhi)

# list lambdas
lambdas = list(np.arange(-13, 9.5, 0.5))

# take SVD of testPhi

'''
svd = np.linalg.svd(trainPhi)
U = svd[0]
S = svd[1]
V = svd[2]
S_Lambda = ComputeS_Lambda(S, 1.0)
'''

print(U.shape)
print(trainY.shape)
print(S.shape)
print(S_Lambda.shape)
print(V.shape)

# U: 6000 x 3080
# trainY: 6000 x 1
# S: 3080 x 3080
# S_Lambda: 3080 x 3080
# V: 3080 x 3080

'''
foo = np.reshape(np.matmul(U.T, trainY), (-1, 1))
bar = np.matmul(S_Lambda, foo)
fizz = np.matmul(V, bar)
'''
