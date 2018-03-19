import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

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

def Center(a):

	# compute column means
	col_means = np.mean(a, axis=0)

	# subtract col means to center matrix
	out = a - col_means

	return(out)

def Scale(a):

	# compute L2 norms of columns of matrix
	norm = np.sqrt(np.sum(pow(a, 2), axis = 0))

	# divide matrix by column norms
	out = a / norm
	
	return(out)

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

def RidgeRegression(U, S_Lambda, V, Y):
	
	# ridge regression coefficients can be calculated as:
	# V * S_Lambda * U.T * Y_train
	# this is most efficiently computed from right to left
	# due to how numpy does SVD we actually need to use V.T rather than V
	ridge_coeffs = np.matmul(V.T, np.matmul(S_Lambda, np.reshape(np.matmul(U.T, Y), (-1, 1))))
	return(ridge_coeffs)

def ScaleRidgeCoeffs(trainPhi, ridge_coeffs):

	# compute L2 norms of columns of trainPhi
	norm = np.reshape(np.sqrt(np.sum(pow(trainPhi, 2), axis = 0)), (-1,1))

	# divide ridge coefficients by norm
	scaled_ridge_coeffs = ridge_coeffs / norm

	return(scaled_ridge_coeffs)

def ScaledRidgeIntercept(trainY, scaled_ridge_coeffs, trainPhi):

	# calculate mean of Y
	y_bar = np.mean(trainY, axis=0)

	# calculate column-wise means of Phi
	Phi_bar = np.reshape(np.mean(trainPhi, axis = 0), (-1,1))

	# multiply scaled ridge coefficient j with Phi mean j
	# and then sum
	w0_scaled = y_bar - np.sum(Phi_bar * scaled_ridge_coeffs, axis = 0)

	return(w0_scaled)

def RidgeErrors(testY, testPhi, w0_scaled, scaled_ridge_coeffs):
	
	n_test = testPhi.shape[0]

	testY = np.reshape(testY, (-1,1))

	ones = np.reshape(np.ones(n_test), (-1,1))
	
	foo = np.reshape(np.matmul(ones, w0_scaled), (-1,1))

	bar = np.matmul(testPhi, scaled_ridge_coeffs)
	
	fizz = testY - foo - bar

	buzz = np.sum(pow(foo, 2), axis = 0)
	
	out = (1 / n_test) * buzz

	return(out)

def RunRidgeAnalysis():
	
	# read data
	X = np.genfromtxt('hw3_X.csv', delimiter = ',')
	Y = np.genfromtxt('hw3_Y.csv', delimiter = ',')
	
	# generate feature matrix Phi
	poly = preprocessing.PolynomialFeatures(2)
	Phi = poly.fit_transform(X)
	
	# delete first column of Phi (which is all 1's)
	Phi = np.delete(Phi, 0, 1)
	
	# segregate into training and test sets
	
	# training set
	trainPhi = GenerateTrain(Phi, 4)
	trainY = GenerateTrain(Y, 4)
	
	# test set
	testPhi = GenerateTest(Phi, 4)
	testY = GenerateTest(Y, 4)
	
	# center and scale training sets
	trainPhi_scaled_centered = Scale(Center(trainPhi))
	trainY_centered = Center(trainY)
	
	# take SVD 
	U, S, V = np.linalg.svd(trainPhi_scaled_centered, full_matrices = False)

	# generate different regularization parameters for ridge regression
	alst = list(np.arange(-13, 9.5, 0.5))
	lambdas = []
	for i in range(0, len(alst)):
		lambdas.append(pow(2, alst[i]))
		print(lambdas)

	print("Lambda",",", "error", sep='')

	# for every tested regularization parameter...
	for Lambda in lambdas:

		# compute S_Lambda
		S_Lambda = ComputeS_Lambda(S, Lambda)
		
		# calculate ridge coeffs
		w = RidgeRegression(U, S_Lambda, V, trainY_centered)
		
		# re-scale ridge coeffs
		w_scaled = ScaleRidgeCoeffs(trainPhi, w)
		
		# calculate w0_scaled
		w0_scaled = ScaledRidgeIntercept(trainY, w_scaled, trainPhi)
		
		# calculate error
		ridge_error = RidgeErrors(testY, testPhi, w0_scaled, w_scaled)
		print(Lambda, ",", ridge_error[0], sep='')

RunRidgeAnalysis()

def RunLassoAnalysis():

	# read data
	X = np.genfromtxt('hw3_X.csv', delimiter = ',')
	Y = np.genfromtxt('hw3_Y.csv', delimiter = ',')
	
	# generate feature matrix Phi
	poly = preprocessing.PolynomialFeatures(2)

	# segregate into training and test sets
	
	# training set
	trainPhi = GenerateTrain(Phi, 4)
	trainY = GenerateTrain(Y, 4)
	
	# test set
	testPhi = GenerateTest(Phi, 4)
	testY = GenerateTest(Y, 4)

	# generate list of regularization parameters to try
	lambdas = list(np.sqrt(log(3080)/6000) * np.arange(-13, 1.5, 0.5))

	lasso = linear_model.Lasso(alpha=1.0)

	fit = lasso.fit(trainPhi, trainY)
