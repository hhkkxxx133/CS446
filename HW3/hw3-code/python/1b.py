from __future__ import print_function
import numpy as np 
import math
from gen import gen

def testing(x, y, w, theta):
	accurate = 0
	i = 0
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y_predict * y[i] > 0:
			accurate += 1
		i += 1
	return accurate

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
	x_new = np.zeros((5000,n+1))
	for row in range(5000):
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



############################################
l = 10
m = 100
n = 500 # or 1000

######################
###### uncomment these for generate the data first
# (y,x) = gen(l, m, n, 50000, False)
# sample = np.random.randint(50000, size = 10000)
# y1 = y[sample[0:5000]]
# x1 = x[sample[0:5000],:]
# y2 = y[sample[5001:10000]]
# x2 = x[sample[5001:10000],:]

# np.save('y11', y1)
# np.save('x11', x1)
# np.save('y22', y2)
# np.save('x22', x2)

###### comment these when generate the data first
x1 = np.load('x11.npy')
y1 = np.load('y11.npy')
x2 = np.load('x22.npy')
y2 = np.load('y22.npy')

# parameter tuning
print('>> Parameter Tuning')
w = np.zeros(n)
theta = 0
for i in range(20):
	w1, theta1 = Perceptron(x1, y1, w, theta, 1)
print('> Perceptron Accuracy: ' + str(testing(x2, y2, w1, theta1)))

w = np.zeros(n)
theta = 0
for i in range(20):
	w2, theta2 = Perceptron_with_margin(x1, y1, w, theta, 0.005, 1)
print('> Perceptron_with_Margin Accuracy: ' + str(testing(x2, y2, w2, theta2)))

w = np.ones(n)
theta = -n
for i in range(20):
	w3, theta3 = Winnow(x1, y1, w, theta, 1.1)
print('> Winnow Accuracy: ' + str(testing(x2, y2, w3, theta3)))

w = np.ones(n)
theta = -n
for i in range(20):
	w4, theta4 = Winnow_with_margin(x1, y1, w, theta, 1.1, 2.0)
print('> Winnow_with_Margin Accuracy: ' + str(testing(x2, y2, w4, theta4)))

weight = np.zeros(n+1)
for i in range(20):
	w5, theta5 = AdaGrad(x1, y1, weight, 0.25)
print('> AdaGrad Accuracy: ' + str(testing(x2, y2, w5, theta5)))


