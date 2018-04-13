import numpy as np
import itertools as it
from sklearn.linear_model import LinearRegression

def powerset(iterable):
	s = list(iterable)
	return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

def RunSimulation():
	
	# read data
	X_train = np.genfromtxt("Xtrain.csv", delimiter = ",")
	X_test = np.genfromtxt("Xtest.csv", delimiter = ",")
	Y_train = np.genfromtxt("Ytrain.csv", delimiter = ",")
	Y_test = np.genfromtxt("Ytest.csv", delimiter = ",")

	# get D and n_train
	n_train = np.shape(X_train)[0]
	D = np.shape(X_train)[1]

	# combine training data into single matrix
	Y_train = np.reshape(Y_train, (n_train, 1))
	mat = np.hstack((X_train, Y_train))

	# list of all possible feature combinations
	fl = list(powerset(np.arange(0, 15, 1)))

	# dont need empty set
	fl.pop(0)

	# skl doesn't have a linear least squares classifier
	# but it does have a ridge classifier
	# and ridge with regularization parameter = 0 is equivalent to linear least squares
	L = LinearRegression()

	# try every feature combination
	for f in fl:

		# get various n
		n_fold_train = int(4*(n_train / 5))
		n_fold_test = int(n_train / 5)

		# get number of features
		D_f = np.shape(f)[0]

		# indices of selected features, and y
		f = np.append(f, 15)

		# sub-select those indices
		mat_f = mat[:, f]

		# break data into 5 folds
		folds = np.split(mat_f, 5)

		# empty list to which we will append CV errors for this model
		err_lst = []

		# 5-fold cross validation
		for fold in folds:
			
			# this fold is for testing
			test_X = fold[:, 0:D_f]
			test_Y = fold[:, D_f]
			
			# skl is fussy about Y's shape
			test_Y = np.reshape(test_Y, (n_fold_test, ))
			
			# everything but this fold is for training
			train = [x for x in folds if not np.array_equal(x, fold)]
			train = np.concatenate(train)
			train_X = train[:, 0:D_f]
			train_Y = np.reshape(train[:, D_f], (n_fold_train, ))

			# fit model
			L.fit(train_X, train_Y)

			# predict based on test_X
			pred = np.sign(L.predict(test_X))

			# calculate error
			err = np.sum((0.5 * abs(test_Y - pred))) / n_fold_test

			# store this fold's CV in list
			err_lst.append(err)

		# take average of errors across folds
		avg_err = sum(err_lst)/5

		# print model and avg err
		print(f, avg_err, sep = ",")

def TestModel():

	# read data
	X_train = np.genfromtxt("Xtrain.csv", delimiter = ",")
	X_test = np.genfromtxt("Xtest.csv", delimiter = ",")
	Y_train = np.genfromtxt("Ytrain.csv", delimiter = ",")
	Y_test = np.genfromtxt("Ytest.csv", delimiter = ",")

	# get n_test
	n_test = np.shape(X_test)[0]

	# narrow to only columns 2, 3, 9, 10
	X_train = X_train[:, [2, 3, 9, 10]]
	X_test = X_test[:, [2, 3, 9, 10]]

	# set up classifier
	L = LinearRegression()

	# train model
	L.fit(X_train, Y_train)

	# predict based on test_X
	pred = np.sign(L.predict(X_test))

	# calculate error
	err = np.sum((0.5 * abs(Y_test - pred))) / n_test

	print(err)

#RunSimulation()
#TestModel()
