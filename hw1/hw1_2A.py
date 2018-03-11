# Tom Wallace
# STAT 672
# Spring 2018
# Homework 1, Problem 2A

import numpy as np

# GenerateXPlusVector(d)
#
# Arguments: d (integer, d >= 1)
#
# Description: generate vector of length d populated with N(5/sqrt(d), 4) Gaussians 
#
# Returns: numpy vector
#
def GenerateXPlusVector(d):

	vector = np.empty(d)

	for i in range(0, d):

		vector[i] = np.random.normal(5/np.sqrt(d), 4)
	
	return(vector)

# GenerateXMinusVector(d)
#
# Arguments: d (integer, d >= 1)
#
# Description: generate vector of length d populated with N(-5/sqrt(d), 1) Gaussians 
#
# Returns: numpy vector
#
def GenerateXMinusVector(d):

	vector = np.empty(d)

	for i in range(0, d):

		vector[i] = np.random.normal(-5/np.sqrt(d), 1)

	return(vector)


# ComputeError(x_plus, x_minus, z)
#
# Arguments: x_plus (numpy vector)
#            x_minus (numpy vector)
#            z (numpy vector)
#
# Description: 
#              - compute ||X_+ - Z||_2^2. Call this norm1.
#              - compute ||X_- - Z||_2^2. Call this norm2.
#              - if A >= B, return 1
#              - else, return 0
#
# Returns: 0 | 1
#
def ComputeError(x_plus, x_minus, z):

	norm1 = np.linalg.norm((x_plus - z), ord=2)
	norm2 = np.linalg.norm((x_minus - z), ord=2) 

	if(norm1 > norm2):
		return(1)
	else:
		return(0)


# RunSimulation(n, d)
#
# Arguments: n (integer, n >= 1)
#            d (integer, d >= 1)
#
# Description: conducts n iterations of generating random vectors and comparing
# them as per above functions.
#
# Returns: nothing (prints to stdout)
#
def RunSimulation(n, d):

	frequency = 0

	for i in range(0, n):
		
		x_plus = GenerateXPlusVector(d)
		z = GenerateXPlusVector(d)
		x_minus = GenerateXMinusVector(d)

		frequency += ComputeError(x_plus, x_minus, z)

	print(d, frequency)


# Run simulation for different d
#
print("d", "occurences of event")
for d in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
	RunSimulation(1000, d)
