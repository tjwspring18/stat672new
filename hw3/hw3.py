'''
Tom Wallace <twalla11@masonlive.gmu.edu>
STAT 672
Spring 2017
HW #3
'''

import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

def GenerateTest(a, i):

	# get number of rows in a
	nrow = a.shape[0]

	# select every i'th row as part of test
	train = a[range(i-1, nrow+1, i)]

	return(train)

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

def ScaleCoeffs(trainPhi, coeffs):

	# compute L2 norms of columns of trainPhi
	norm = np.reshape(np.sqrt(np.sum(pow(trainPhi, 2), axis = 0)), (-1,1))

	# divide coefficients by norm
	scaled_coeffs = coeffs / norm

	return(scaled_coeffs)

def ScaledIntercept(trainY, scaled_coeffs, trainPhi):

	# calculate mean of Y
	y_bar = np.mean(trainY, axis=0)

	# calculate column-wise means of Phi
	Phi_bar = np.reshape(np.mean(trainPhi, axis = 0), (-1,1))

	# multiply scaled coefficient j with Phi mean j
	# and then sum
	w0_scaled = y_bar - np.sum(Phi_bar * scaled_coeffs, axis = 0)

	return(w0_scaled)

def Errors(testY, testPhi, w0_scaled, scaled_coeffs):
	
	n_test = testPhi.shape[0]

	testY = np.reshape(testY, (-1,1))

	ones = np.reshape(np.ones(n_test), (-1,1))
	
	foo = np.reshape(np.matmul(ones, w0_scaled), (-1,1))

	bar = np.matmul(testPhi, scaled_coeffs)
	
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
	

	# for every tested regularization parameter...
	for Lambda in lambdas:

		# compute S_Lambda
		S_Lambda = ComputeS_Lambda(S, Lambda)
		
		# calculate ridge coeffs
		w = RidgeRegression(U, S_Lambda, V, trainY_centered)
		
		# re-scale ridge coeffs
		w_scaled = ScaleCoeffs(trainPhi, w)
		
		# calculate w0_scaled
		w0_scaled = ScaledIntercept(trainY, w_scaled, trainPhi)
		
		# calculate error
		ridge_error = Errors(testY, testPhi, w0_scaled, w_scaled)

		# output results
		print("ridge,", Lambda, ",", ridge_error[0], sep='')

def RunLassoAnalysis():

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

	# generate list of regularization parameters to try

	alst = list(np.arange(-13, 9.5, 0.5))
	lambdas = []
	for i in range(0, len(alst)):
		lambdas.append(np.sqrt(np.log(3080)/6000) * pow(2, alst[i]))
	lambdas.reverse() #decreasing order

	# for every tested regularization parameter...
	for Lambda in lambdas:
		
		# fit lasso model with that regularization parameter
		# note warm start
		# note not fitted with intercept (since already centered and scaled)
		lasso = linear_model.Lasso(alpha=Lambda, warm_start = True, fit_intercept = False)
		fit = lasso.fit(trainPhi_scaled_centered, trainY_centered)

		# get coefficients
		w = np.reshape(fit.coef_, (-1,1))

		# re-scale coeffs
		w_scaled = ScaleCoeffs(trainPhi, w)
		
		# calculate w0_scaled
		w0_scaled = ScaledIntercept(trainY, w_scaled, trainPhi)
		
		# calculate error
		lasso_error = Errors(testY, testPhi, w0_scaled, w_scaled)

		# output results
		print("lasso,", Lambda, ",", lasso_error[0], sep='')

RunRidgeAnalysis()
RunLassoAnalysis()
