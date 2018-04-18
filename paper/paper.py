import numpy as np

X = np.random.randn(int(10e6))
Y = (0.5 * X) + np.random.randn(1)

def GD(X, Y, gamma, epsilon, maxiter):
	n = Y.size
	w = 1 #initial guess
	for i in range(maxiter):
		Yhat = np.dot(X, w)
		update = w - gamma * (np.dot(X.T, Yhat - Y) / n)
		if (abs(w - update) < epsilon): 
			break
		else:
			w = update
	return(w)

def SGD(X, Y, gamma, epsilon, maxiter):
	n = Y.size
	w = 1 #initial guess
	for i in range(maxiter):
		j = np.random.randint(0, n)
		Yhat = w * X[j]
		update = w - gamma * ((Yhat - Y[j])*X[j])
		if (abs(w - update) < epsilon):
			break
		else:
			w = update
	return(w)
#print(GD(X, Y, 0.05, 0.00001, 10000000))
print(SGD(X, Y, 0.05, 0.00001, 10000000))


