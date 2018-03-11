# Tom Wallace
# STAT 672
# Spring 18
# Homework 1, Problem 1B

import numpy as np

# GenerateTheta(d)
#
# Arguments: d (int, d >= 1)
#
# Description: generates vector of dimensionality d filled with uniform random
# variables from surface of d-dimensional sphere. Computed as Z / ||Z||_2 where
# Z is a vector populated with d N(0,1) random variables.
#
# Returns: numpy vector
#
def GenerateTheta(d):

	gaussians = np.random.randn(d)
	
	normalization = np.linalg.norm(gaussians, 2)

	return(gaussians / normalization)

# GenerateRadius(d)
#
# Arguments: d (int, d >= 1)
# 
# Description: generates scalar between 0 and 1. Calculated as U^(1/d), where U
# is a uniform(0,1) random variable and d is dimensionality.
#
# Returns: real
#
def GenerateRadius(d):

	u = np.random.uniform(0,1)

	r = np.power(u, 1/d)

	return(r)

# PolarSampling(n, d):
#
# Arguments: n (int, n >= 1)
#            d (int, d >= 1) 
#
# Description: generates n random vectors of the d-dimensional unit ball. Does
# so by computing the product of theta and radius.
#
# Returns: nothing (prints random vectors to stdout)
#
def PolarSampling(d, n):

	i = 0

	print("Generating", n, "random vectors of dimensionality", d)

	while(i < n):
		
		theta = GenerateTheta(d)
		r = GenerateRadius(d)
		vector = np.dot(theta, r)
		i += 1
		print(vector)


#create 100 samples with d=10
PolarSampling(10, 100)
