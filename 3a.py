import os
import random
import sys
import numpy as np
from random import shuffle

## constants
inputData = "mnist_2_vs_7/mnist_X_train.dat"
expectedOutputData = "mnist_2_vs_7/mnist_Y_train.dat"
xTestData = "mnist_2_vs_7/mnist_X_test.dat"
yTestData = "mnist_2_vs_7/mnist_Y_test.dat"

learningRate = .1
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

	t = 0
	wGD = np.zeros(len(xfile[0]))
	iterations = 50
	batch = len(xfile)
	training(wGD, xfile, yfile, iterations, batch) # part 3a
	success = predAccu(wGD, testX, testY) # part 3b
#	print (success) # tests accuracy for 3a/b


	n = 256 # randomly choose n=256 data points from X
#	trainedW = SGD(wSGD, n, xfile, yfile) # 3c: function to implement SGD on logistic regression
	np.random.shuffle(xfile) # shuffle input/training vectors/data to stochastically choose N data points
	wSGD = np.zeros(len(xfile[0])) # reinitializes weight
	trainedW = trainingSGD(wSGD, xfile, yfile)
	successSGD = predAccu(trainedW, testX, testY)
#	print(trainedW)
	print (successSGD)
#	predAccuSGD(w, testX, testY)

def trainingSGD(wSGD, xfile, yfile):
	batch = 256
	iterations = 500
	training(wSGD, xfile, yfile, iterations, batch) # part 3a
	return wSGD

def training(wGD, xfile, yfile, iterations, batch): 
	for i in range(iterations): # random big number
		grad = summation(wGD, xfile, yfile, batch)/len(xfile) # compute the gradient
		wGD += learningRate * grad
	return wGD # updated weight

def summation(wGD, xfile, yfile, batch): # summation in gradient
	sum = 0.0
	for j in range(batch): # i should go from 1
		sum += xfile[j] * yfile[j] / (1 + np.exp(yfile[j] * np.dot(np.transpose(wGD), xfile[j])))
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






