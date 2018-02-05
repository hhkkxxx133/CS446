from __future__ import print_function
import numpy as np 
import math
import matplotlib.pyplot as plt
from gen import gen

def Perceptron(x, y, r):
	w = np.zeros(n)
	theta = 0
	i = 0
	error = 0
	mistakes = []
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y[i]*y_predict <= 0:
			error += 1
			w += np.multiply(r*y[i],x_curr)
			theta += y[i]
		mistakes.append(error)
		i += 1
	return mistakes

def Perceptron_with_margin(x, y, r, margin):
	w = np.zeros(n)
	theta = 0
	i = 0
	error = 0
	mistakes = []
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y[i]*y_predict <= 0:
			error += 1
		if y[i]*y_predict < margin:
			w += np.multiply(r*y[i],x_curr)
			theta += r*y[i]
		mistakes.append(error)
		i += 1
	return mistakes

def Winnow(x, y, alpha):
	w = np.ones(n)
	theta = -n
	i = 0
	error = 0
	mistakes = []
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y[i]*y_predict <= 0:
			error += 1
			for j in range(x_curr.size):
				w[j] = w[j] * pow(alpha, y[i]*x_curr[j])
		mistakes.append(error)
		i += 1
	return mistakes

def Winnow_with_margin(x, y, alpha, margin):
	w = np.ones(n)
	theta = -n
	i = 0
	error = 0
	mistakes = []
	for x_curr in x:
		y_predict = np.dot(x_curr,w) + theta
		if y[i]*y_predict <= 0:
			error += 1
		if y[i]*y_predict < margin:
			for j in range(x_curr.size):
				w[j] = w[j] * pow(alpha, y[i]*x_curr[j])
		mistakes.append(error)
		i += 1
	return mistakes

def AdaGrad(x, y, r):
	weight = np.zeros(n+1)
	error = 0
	mistakes = []
	x_new = np.zeros((50000,n+1))
	for row in range(50000):
		x_new[row,:] = np.concatenate((x[row,:], [1]), axis = 0)
	gt = np.zeros(n+1)
	i = 0
	for x_curr in x_new:
		y_predict = np.dot(x_curr,weight)
		if y[i] * y_predict <= 0:
			error += 1
		if y[i] * y_predict <= 1:
			# print('yes if')
			gti = np.multiply(-y[i], x_curr)
			gt += np.multiply(gti, gti)
			for j in range(weight.size):
				if gt[j] != 0:
					weight[j] += r*y[i]*x_curr[j]/math.sqrt(gt[j])
		mistakes.append(error)
		i += 1
	return mistakes


############################################
l = 10
m = 100
n = 1000 # 500 or  1000

# mistakes tracking
(y,x) = gen(l, m, n, 50000, False)

mistakes_all = []
mistakes_all.append(Perceptron(x, y, 1))
mistakes_all.append(Perceptron_with_margin(x, y, 0.005, 1))
mistakes_all.append(Winnow(x, y, 1.1))
mistakes_all.append(Winnow_with_margin(x, y, 1.1, 2.0))
mistakes_all.append(AdaGrad(x, y, 0.25))

plt.plot(range(50000), mistakes_all[0], label = 'Perceptron')
plt.plot(range(50000), mistakes_all[1], label = 'Perceptron w/ margin')
plt.plot(range(50000), mistakes_all[2], label = 'Winnow')
plt.plot(range(50000), mistakes_all[3], label = 'Winnow w/ margin')
plt.plot(range(50000), mistakes_all[4], label = 'AdaGrad')
plt.title('Mistakes for n = 1000')
plt.xlabel('N')
plt.ylabel('W')
plt.legend(loc = 2)
plt.show()






