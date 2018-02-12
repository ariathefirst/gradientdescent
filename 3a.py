import os
import random
import sys
import numpy as np
from random import shuffle
from numpy import random as rd

## constants
inputData = "mnist_2_vs_7/mnist_X_train.dat"
expectedOutputData = "mnist_2_vs_7/mnist_Y_train.dat"
xTestData = "mnist_2_vs_7/mnist_X_test.dat"
yTestData = "mnist_2_vs_7/mnist_Y_test.dat"

xfileSize = 10000

output = []
trainedOut = []
predictedVal = []
err = []

def main():
	xfile = np.loadtxt(inputData)
	yfile = np.loadtxt(expectedOutputData)
	testX = np.loadtxt(xTestData)
	testY = np.loadtxt(yTestData)
	prob = 0

	# 3a/3b
	# t = 0
	# learningRate = .1
	# wGD = np.zeros(len(xfile[0]))
	# iterationsGD = 50
	# batchGD = len(xfile)
	# trainedWGD = training(learningRate, wGD, xfile, yfile, iterationsGD, batchGD) # part 3a
	# success = predAccu(wGD, testX, testY) # part 3b
	# print (success) # tests accuracy for 3a/b
	# objFun(xfile, yfile, trainedWGD)

	# 3c
# 	learningRate = .00001
# 	iterationsSGD = 350
# 	batchSGD = 256 # randomly choose n=256 data points from X
# 	np.random.shuffle(xfile) # shuffle input/training vectors/data to stochastically choose N data points
# 	wSGD = np.zeros(len(xfile[0])) # reinitializes weight
# 	trainedW = training(learningRate, wSGD, xfile, yfile, iterationsSGD, batchSGD, prob)
# 	successSGD = predAccu(trainedW, testX, testY)
# #	print(trainedW)
# 	print (successSGD)

	# #SGD with decaying step size, 3d
	prob = "3d"
	learningRate = .00001
	iterationsSGD = 350
	batchSGD = 256 # randomly choose n=256 data points from X
	np.random.shuffle(xfile) # shuffle input/training vectors/data to stochastically choose N data points
	wDecaySGD = np.zeros(len(xfile[0])) # reinitializes weight
	trainedW = training(learningRate, wDecaySGD, xfile, yfile, iterationsSGD, batchSGD, prob)
	successSGD = predAccu(trainedW, testX, testY)
	print(successSGD)	


def objFun(xfile, yfile, w):
	for i in range(len(xfile)):
		comparison = 1 - yfile[i] * np.dot(np.transpose(w), xfile[i])
		En = max(0, comparison)
		if En != 0:
			print(En)
	return

def training(learningRate, w, xfile, yfile, iterations, batch, prob): 
	for i in range(iterations): # random big number
		if batch < len(yfile):	# SGD case	
			np.random.shuffle(xfile) # shuffle input/training vectors/data to stochastically choose N data points
			if prob == '3d' and i % 50 == 0:
				learningRate * .5
		grad = summation(i, w, xfile, yfile, batch)/batch # compute the gradient
		w += learningRate * grad
	return w # updated weight

def summation(i, w, xfile, yfile, batch): # summation in gradient
	sum = 0.0
	for j in range(batch): # i should go from 1
		sum += xfile[j] * yfile[j] / (1 + np.exp(yfile[j] * np.dot(np.transpose(w), xfile[j])))
	return sum

def predAccu(w, testX, testY): # part 3b, sum of err divided by xfilelen
	hit = 0
	for i in range(len(testX)):
		trainedY = np.dot(np.transpose(w), testX[i])
		if trainedY >= 0:
			trainedY = 1
		elif trainedY < 0:
			trainedY = -1
		if trainedY == testY[i]:
			hit += 1
	hits = hit / len(testX)
	return hits

if __name__ == "__main__":
    main()


# def SGD(wSGD, n, xfile, yfile):
# 	np.random.shuffle(xfile) # shuffle input/training vectors/data to stochastically choose N data points
# 	grad = 0.0
# 	for i in range(500): # sample n samples 500 times
# 		for i in range(n):
# 			ranX = random.choice(xfile)
# 			ranY = yfile[i]
# 			grad = -1 * ranY * ranX / (1 + np.exp(ranX * np.dot(np.transpose(wSGD), ranX)))
# 			wSGD -= learningRate * grad
# 	return wSGD






