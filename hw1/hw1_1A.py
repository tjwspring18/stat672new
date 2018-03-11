# Tom Wallace
# STAT 672
# Spring 2018
# Homework 1, Problem 1A

import numpy as np

# UnitBallRejectionSampling(d, n)
#
# Arguments: d (int, d >= 1)
#            n (int, n >= 1)
#
# Description: draw n random vectors from d-dimensional unit ball. Does so by
# generating a candidate vector, each coordinate of which is a random
# draw from a uniform (-1,1) distribution, and rejecting any candidate vector
# with a Euclidean norm >= 1
#
def UnitBallRejectionSampling(d, n):
	
	n_candidates = 0
	n_accepted = 0

	print("Generating", n, "random vectors of dimensionality", d, "using rejection sampling")
	
	while(n_accepted < n):

		#generate empty vector of length d
		vector = np.empty(d)

		#fill with (-1,1) random variables
		for i in range(0, d):
			vector[i] = np.random.uniform(-1,1)

		#update n_candidates
		n_candidates += 1

		#take Euclidean norm
		euclidean_norm = np.linalg.norm(vector, ord=2) 
		
		#accept only if Euclidean norm <= 1
		if(euclidean_norm <= 1):
			print(vector)
			n_accepted += 1
		else:
			pass

	#calculate and print observed probability of acceptance
	prob_accept = n_accepted / n_candidates
	print("Probability of acceptance: ", prob_accept)


#generate 100 samples with d=10
UnitBallRejectionSampling(10,100)
