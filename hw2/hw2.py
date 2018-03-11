import numpy as np
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Tom Wallace
# STAT 672
# Spring 2018
# Homework 2

#####                     #####
#####     Problem 1 C     #####
#####                     #####

# use np.random.uniform() to generate X ~ uniform(0,1)

# function to generate Y given X
def GenerateYGivenX(x):

	r = np.random.uniform()

	if(x < 0.2 or x > 0.8):

		if(r <= 0.9):
			y = 1
		else:
			y = 0
	else:
		if(r <= 0.2):
			y = 1
		else:
			y = 0

	return(y)

# vectorized version of Y-generating function
v_GenerateYGivenX = np.vectorize(GenerateYGivenX)

# run many iterations of the above functions
# plot results to verify correctness

# set random seed
np.random.seed(888)

# generate 10k x
X = np.random.rand(10000)

# generate 10k y | x
Y = v_GenerateYGivenX(X)

# round X to 2 digits
v_round = np.vectorize(np.round)
X = v_round(X, 2)

# count number of y=1 for every x
histX = np.arange(0, 1, 0.01)

histY = np.zeros(len(histX))

for i in range(0, len(histX)):
	for j in range(0, len(X)):
		if( histX[i] == X[j] and Y[j] == 1 ):
			histY[i] = histY[i]+1
# bin X's into 10 bins

histX2 = np.arange(0, 1, 0.1)
histY2 = np.zeros(10)

histY2[0] = sum(histY[0:9])
histY2[1] = sum(histY[10:19])
histY2[2] = sum(histY[20:29])
histY2[3] = sum(histY[30:39])
histY2[4] = sum(histY[40:49])
histY2[5] = sum(histY[50:59])
histY2[6] = sum(histY[60:69])
histY2[7] = sum(histY[70:79])
histY2[8] = sum(histY[80:89])
histY2[9] = sum(histY[90:99])

# plot
plt.bar(x=histX2, height=histY2, width=0.155555)
plt.title("10,000 Simulations of Y|X")
plt.xlabel("X")
plt.ylabel("N(Y=1|X=x)")
plt.savefig("hist.png")

# code to generate feature matrix for arbitrary dimensions d and number of
# observations n
def GenerateFeatureMatrix(n, d):

	#fill vector of ones
	ones = np.ones(n)

	#create vector of uniform(0,1) random variables
	X = np.random.rand(n)

	#column bind these two vectors into a matrix
	mat = np.column_stack((ones, X))

	#create vectorized version of pow function
	vexp = np.vectorize(pow)

	#raise vector X to the 2, 3...d power
	#column bind this new vector with our matrix
	for i in range(2, d+1):
		vec = vexp(X,i)
		mat = np.column_stack((mat, vec))

	#return the matrix
	return(mat)

#####                     #####
#####     Problem 1 D     #####
#####                     #####

# d = 1

# generate X
X1 = GenerateFeatureMatrix(100, 1)

# remove first column of 1's
X1 = np.delete(X1, 0, 1)

# generate Y given X
y1 = v_GenerateYGivenX(X1[:,0])

# fit model 
lr1 = linear_model.LogisticRegression(penalty = "l2", fit_intercept=True, C=1e5)
lr1.fit(X1, y1)

# generate predictions
#p1 = lr1.predict(X1[:,])

# d = 2
# steps are the same as above, so omitting comments
X2 = GenerateFeatureMatrix(100, 2)
X2 = np.delete(X2, 0, 1)
y2= v_GenerateYGivenX(X2[:,0])
lr2 = linear_model.LogisticRegression(penalty = "l2", fit_intercept=True, C=1e5)
lr2.fit(X2, y2)
#p2 = lr2.predict(X2[:,])

# d = 5
# steps are the same as above, so omitting comments
X5 = GenerateFeatureMatrix(100, 5)
X5 = np.delete(X5, 0, 1)
y5= v_GenerateYGivenX(X5[:,0])
lr5 = linear_model.LogisticRegression(penalty = "l2", fit_intercept=True, C=1e5)
lr5.fit(X5, y5)
#p5 = lr5.predict(X5[:,])

# use each fitted model to predict based on fixed set of X's
# these results are presented in Table 1 of my LaTeX writeup
to_predict_1 = [[0], [0.25], [0.4], [0.75], [1]]
to_predict_2 = [[0, 0], 
		[0.25, pow(0.25, 2)],
		[0.50, pow(0.50, 2)],
		[0.75, pow(0.75, 2)],
		[1.00, pow(1, 2)]]
to_predict_5 = [[0,0,0,0,0], 
		[0.25, pow(0.25, 2), pow(0.25, 3), pow(0.25, 4), pow(0.25, 5)],
		[0.50, pow(0.50, 2), pow(0.50, 3), pow(0.50, 4), pow(0.50, 5)],
		[0.75, pow(0.75, 2), pow(0.75, 3), pow(0.75, 4), pow(0.75, 5)],
		[1.00, pow(1.00, 2), pow(1.00, 3), pow(1.00, 4), pow(1.00, 5)],
		]
print("Problem 1 C")
print("Predictions of Y|X for d=1, 2, 5 and X=0, 0.25, 0.5, 0.75, 1")
print(lr1.predict(to_predict_1))
print(lr2.predict(to_predict_2))
print(lr5.predict(to_predict_5))



#####                     #####
#####     Problem 1 E     #####
#####                     #####

def RunSimulation(n, d, n_iter):

	print("Running simulation for n = ",n,", d =",d,", and iterations =", n_iter)

	emp_error = 0
	gen_error = 0

	for i in range(0, n_iter):

		# generate two independent feature matrices
		D1 = GenerateFeatureMatrix(n, d)
		D2 = GenerateFeatureMatrix(n, d)
		
		#remove first column of 1's
		D1 = np.delete(D1, 0, 1)
		D2 = np.delete(D2, 0, 1)
		
		# generate corresponding Y | X
		y1 = v_GenerateYGivenX(D1[:,0])
		y2 = v_GenerateYGivenX(D2[:,0])
		
		# train logistic regression model on D1 only
		model = linear_model.LogisticRegression(penalty = "l2", fit_intercept=True, C=1e5)
		model.fit(D1, y1)
		
		# calculate error rates
		emp_error += 1-metrics.accuracy_score(y1, model.predict(D1))
		gen_error += 1-metrics.accuracy_score(y2, model.predict(D2))
	
	#average over all iterations
	emp_error = emp_error / n_iter
	gen_error = gen_error / n_iter

	#print results
	print("Avg. emp_error =", emp_error)
	print("Avg. gen_error =", gen_error)

# run simulation for varying n and d

print(" ")
print("Problem 1 E")

N = [10, 20, 50, 100, 200, 500, 1000]

for n in N:
	for d in [1, 2, 5]:
		RunSimulation(n, d, 50)


#plotted results separately in R - see LaTeX PDF submission for charts
