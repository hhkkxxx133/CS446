from __future__ import print_function
import numpy as np 
import math
import matplotlib.pyplot as plt
from gen import gen

def testing(x, y, w, theta):
	accurate = 0
	i = 0
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y_predict * y[i] > 0:
			accurate += 1
		i += 1
	return float(accurate)/float(i)

def Perceptron(x, y, w, theta, r):
	i = 0
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y[i]*y_predict <= 0:
			w += np.multiply(r*y[i],x_curr)
			theta += y[i]
		i += 1
	return w, theta

def Perceptron_with_margin(x, y, w, theta, r, margin):
	i = 0
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y[i]*y_predict < margin:
			w += np.multiply(r*y[i],x_curr)
			theta += r*y[i]
		i += 1
	return w, theta

def Winnow(x, y, w, theta, alpha):
	i = 0
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y[i]*y_predict <= 0:
			for j in range(x_curr.size):
				w[j] = w[j] * pow(alpha, y[i]*x_curr[j])
		i += 1
	return w, theta

def Winnow_with_margin(x, y, w, theta, alpha, margin):
	i = 0
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y[i]*y_predict < margin:
			for j in range(x_curr.size):
				w[j] = w[j] * pow(alpha, y[i]*x_curr[j])
		i += 1
	return w, theta

def AdaGrad(x, y, weight, r):
	x_new = np.zeros((50000,n+1))
	for row in range(50000):
		x_new[row,:] = np.concatenate((x[row,:], [1]), axis = 0)
	gt = np.zeros(n+1)
	i = 0
	for x_curr in x_new:
		y_predict = np.dot(x_curr,weight)
		if y[i] * y_predict <= 1:
			gti = np.multiply(-y[i], x_curr)
			gt += np.multiply(gti, gti)
			for j in range(weight.size):
				if gt[j] != 0:
					weight[j] += r*y[i]*x_curr[j]/math.sqrt(gt[j])
		i += 1
	w = weight[0:n]
	theta = weight[n]
	return w, theta


##################################################
## P1
l = 10
m = 1000
n = 1000

# # generate training and testing data
# (train_y, train_x) = gen(l, m, n, 50000, True)
# (test_y, test_x) = gen(l, m, n, 10000, False)

# # save to local disk
# np.save('trainy_file', train_y)
# np.save('trainx_file', train_x)
# np.save('testy_file', test_y)
# np.save('testx_file', test_x)

## read from local disk
x_train = np.load('trainx_file.npy')
y_train = np.load('trainy_file.npy')
x_test = np.load('testx_file.npy')
y_test = np.load('testy_file.npy')

w = np.zeros(n)
theta = 0
for i in range(20):
	w1, theta1 = Perceptron(x_train, y_train, w, theta, 1)
print('> Perceptron Accuracy: ' + str(testing(x_test, y_test, w1, theta1)))

w = np.zeros(n)
theta = 0
for i in range(20):
	w2, theta2 = Perceptron_with_margin(x_train, y_train, w, theta, 0.03, 1)
print('> Perceptron_with_Margin Accuracy: ' + str(testing(x_test, y_test, w2, theta2)))

w = np.ones(n)
theta = -n
for i in range(20):
	w3, theta3 = Winnow(x_train, y_train, w, theta, 1.1)
print('> Winnow Accuracy: ' + str(testing(x_test, y_test, w3, theta3)))

w = np.ones(n)
theta = -n
for i in range(20):
	w4, theta4 = Winnow_with_margin(x_train, y_train, w, theta, 1.1, 0.006)
print('> Winnow_with_Margin Accuracy: ' + str(testing(x_test, y_test, w4, theta4)))

weight = np.zeros(n+1)
for i in range(20):
	w5, theta5 = AdaGrad(x_train, y_train, weight, 1.5)
print('> AdaGrad Accuracy: ' + str(testing(x_test, y_test, w5, theta5)))

