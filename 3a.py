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
	training(wGD, xfile, yfile) # part 3a
#	print(w)

	success = predAccu(wGD, testX, testY) # part 3b
#	print (success) # tests accuracy for 3a/b

	n = 50 # randomly choose n=500 data points from X
	wSGD = np.zeros(len(xfile[0])) # reinitializes weight
	trainedW = SGD(wSGD, n, xfile, yfile) # 3c: function to implement SGD on logistic regression
	successSGD = predAccu(trainedW, testX, testY)
#	print(trainedW)
	print (successSGD)
#	predAccuSGD(w, testX, testY)

def SGD(wSGD, n, xfile, yfile):
	shuffle(xfile) # shuffle input/training vectors/data to stochastically choose N data points
	shuffle(yfile) 
	grad = 0;
	for i in range(n):
		ranX = random.choice(xfile)
		ranY = random.choice(yfile)
		grad = -1 * ranY * ranX / (1 + np.exp(ranX * np.dot(np.transpose(wSGD), ranX)))
		wSGD -= learningRate * grad
	return wSGD



def training(wGD, xfile, yfile): 
	for i in range(50): # random big number
		grad = summation(wGD, xfile, yfile)/len(xfile) # compute the gradient
		wGD += learningRate * grad
	return wGD # updated weight

def summation(wGD, xfile, yfile): # summation in gradient
	sum = 0
	for j in range(xfileSize): # i should go from 1
		sum += xfile[j] * yfile[j] / (1 + np.exp(yfile[j] * np.dot(np.transpose(wGD), xfile[j])))
	return sum

def predAccu(w, testX, testY): # part 3b, sum of err divided by xfilelen
	# for i in range(xfileSize):
	# 	output[i] = np.dot(np.transpose(w), xfile[i])
	# 	for j in range(output):
	# 		if trainedOut[j] > 0:
	# 			predictedVal[j] = 1
	# 		elif trainedOut[i]:
	# 			predictedVal[j] = -1
	# for i in range(xfileSize):
	# 	if predictedVal[i] >= 0:
	# 		err[i] = 1
	# 	elif predictedVal[i] < 0:
	# 		err[i] = -1	
	# 	errSum += err[i]
	# accuracy = errSum / xfileSize
	# return accuracy
	hit = 0
	for i in range(len(testX)):
		trainedY = np.dot(np.transpose(w), testX[i])
		if trainedY >= 0:
			trainedY = 1
		elif trainedY < 0:
			trainedY = -1
		if trainedY == testY[i]:
			hit += 1
	hits = float(hit) / float(len(testX))
	return hits

if __name__ == "__main__":
    main()






