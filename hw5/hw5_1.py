# Tom Wallace
# STAT 672
# Spring 2018
# Homework 5
# Part I

import numpy as np
from sklearn.linear_model import LogisticRegression

def RunSimulation(n, D):
	
	# n x D feature matrix
	# X_ij = N(0,1)
	X = np.random.randn(n, D)
	
	# Y = {1, -1, 1, -1...}
	Y = np.ones(n)
	Y[1::2] = Y[1::2] * -1
	Y = np.reshape(Y, (n, 1))

	# c_j = \frac{1}{n} \sum_{i=1}^n X_{ij}Y_i
	C = np.sum(X * Y, 0) / n

	# find indices with largest magnitude
	C = abs(C)
	ind = np.argpartition(C, -5)[-5:]

	# new feature matrix X_s
	X_s = X[:,ind]

	# join X_s and Y into one matrix
	mat = np.hstack((X_s, Y))

	# 10-fold cross validation
	# break data into 10 folds
	folds = np.split(mat, 10)

	lst = []
	# for every fold...
	for fold in folds:

		# this fold is for testing
		test_X = fold[:, 0:5]
		test_Y = fold[:, 5]

		# skl logistic regression is fussy about Y's shape
		test_Y = np.reshape(test_Y, (int(n/10),))

		# everything but this fold is for training
		train = [x for x in folds if not np.array_equal(x, fold)]
		train = np.concatenate(train)
		train_X = train[:, 0:5]
		train_Y = np.reshape(train[:, 5], (int(9 * (n/10)),))

		# train logistic regression 
		L = LogisticRegression()
		L.fit(train_X, train_Y)

		# validate
		err = 1-L.score(test_X, test_Y)

		lst.append(err)
	
	print("10-fold CV error:", sum(lst)/10)

RunSimulation(10000, 100)
