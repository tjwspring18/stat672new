import numpy as np
from sklearn.linear_model import LogisticRegression

# Part 1
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

	# logistic regression
	Y = np.reshape(Y, (n,))
	L = LogisticRegression()
	L.fit(X_s, Y)
	print(L.coef_)
	print(L.intercept_)

RunSimulation(100, 10000)
