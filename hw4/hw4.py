import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

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

def RunAnalysis():

	# read data
	xTrain = np.genfromtxt('spam_Xtrain.csv', delimiter = ',')
	xTest = np.genfromtxt('spam_Xtest.csv', delimiter = ',')
	yTrain = np.genfromtxt('spam_Ytrain.csv', delimiter = ',')
	yTest = np.genfromtxt('spam_Ytest.csv', delimiter = ',')

	# read feature labels
	with open('featurenames', 'r') as f:
		labels = f.readlines()

	# center and scale data
	xTrain_cs = Scale(Center(xTrain))
	xTest_cs = Scale(Center(xTest))

	# regularization parameters to test
	C = pow(10, np.arange(-1, 6.5, 0.5))

	# fit SVM with different reg. parameter
	for c in C:

		print("SVM, regularization parameter =", c)
		print("")

		# initialize SVC
		svm = SVC(c, kernel = "linear")

		# fit on training data
		svm.fit(xTrain_cs, yTrain)

		# number of support vectors
		print("Number of support vectors")
		print(np.sum(svm.n_support_))
		print("")

		# training error
		print("Training error")
		print(1 - svm.score(xTrain_cs, yTrain))
		print("")

		# test error
		print("Test error")
		print(1 - svm.score(xTest_cs, yTest))
		print("")

		# calculate fp and tp
		y_pred = svm.predict(xTest_cs)
		y_true = yTest
		tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
		n_test = np.shape(y_true)[0]

		# false positives
		print("False positive rate, test set")
		print(fp / n_test)
		print("")

		# true positives
		print("True positive rate, test set")
		print(tp / n_test)
		print("")

		# optimal weight vector
		print("Weight vector")
		for i in range(0, len(labels)):
			print(labels[i].rstrip(), svm.coef_[0][i])
		print("")
		print("**************************************************************")
		print("")

RunAnalysis()
