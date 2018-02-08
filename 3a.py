import os
import random
import sys
## pip3 install opencv-python
import numpy as np

%matplotlib inline

## constants
inputDate = "mnist_2_vs_7/mnist_X_train.dat"
expectedOutputData = "mnist_2_vs_7/mnist_Y_train.dat"

learningRate = .1
xfileSize = 10000


def main()
	xfile = numpy.loadtxt(inputDate)
	yfile = numpy.loadtxt(expectedOutputData)
	t = 0
	w = numpy.zeros()
	training(w)

def training(w) 
	for i in range(5000) # random big number
		grad = summation(w)/len(xfile) # compute the gradient
		w += learningRate * grad
	return w # updated weight

def summation(w) # summation in gradient
	sum = 0
	for i in xfileSize - 1 # i should go from 1
		sum += xfile[i] * yfile[i] / (1 + np.exp(yfile[i] * np.dot(np.transpose(w), xfile[i])))
	return sum






