from __future__ import print_function
import numpy as np 
import math
import matplotlib.pyplot as plt
from gen import gen


def Perceptron(x, y, r):
	error = 0
	convergence = 0
	# mistakes = []
	w = np.zeros(n[idx])
	theta = 0
	while convergence < R:
		for k in range(10):
			i = 0
			for x_curr in x:
				y_predict = np.dot(x_curr,w) + theta
				if y[i]*y_predict <= 0:
					error += 1
					convergence = 0
					w += np.multiply(r*y[i],x_curr)
					theta += y[i]
				else:
					convergence += 1
					if convergence >= R :
						break
				# mistakes.append(error)
				i += 1
			if convergence >= R:
				break
	return error

def Perceptron_with_margin(x, y, r, margin):
	error = 0
	# mistakes = []
	convergence = 0
	w = np.zeros(n[idx])
	theta = 0
	while convergence < R:
		for k in range(10):
			i = 0
			for x_curr in x:
				y_predict = np.dot(x_curr,w) + theta
				if y[i]*y_predict <= 0:
					convergence = 0
					error += 1
				else:
					convergence += 1
					if convergence >= R:
						break
				if y[i]*y_predict < margin:
					w += np.multiply(r*y[i],x_curr)
					theta += r*y[i]
				# mistakes.append(error)
				i += 1
			if convergence >= R:
				break
	return error

def Winnow(x, y, alpha):
	error = 0
	# mistakes = []
	convergence = 0
	w = np.ones(n[idx])
	theta = -n[idx]
	while convergence < R:
		for k in range(10):
			i = 0
			for x_curr in x:
				y_predict = np.dot(x_curr,w) + theta
				if y[i]*y_predict <= 0:
					convergence = 0
					error += 1
					for j in range(x_curr.size):
						w[j] = w[j] * pow(alpha, y[i]*x_curr[j])
				else:
					convergence += 1
					if convergence >= R:
						break
				# mistakes.append(error)
				i += 1
			if convergence >= R:
				break
	return error

def Winnow_with_margin(x, y, alpha, margin):
	error = 0
	# mistakes = []
	convergence = 0
	w = np.ones(n[idx])
	theta = -n[idx]
	while convergence < R:
		for k in range(10):
			i = 0
			for x_curr in x:
				y_predict = np.dot(x_curr,w) + theta
				if y[i]*y_predict <= 0:
					convergence = 0
					error += 1
				else:
					convergence += 1 
					if convergence >= R:
						break
				if y[i]*y_predict < margin:
					for j in range(x_curr.size):
						w[j] = w[j] * pow(alpha, y[i]*x_curr[j])
				# mistakes.append(error)
				i += 1
			if convergence >= R:
				break
	return error

def AdaGrad(x, y, r):
	convergence = 0
	error = 0
	# mistakes = []
	weight = np.zeros(n[idx]+1)
	x_new = np.zeros((50000,n[idx]+1))
	for row in range(50000):
		x_new[row,:] = np.concatenate((x[row,:], [1]), axis = 0)
	gt = np.zeros(n[idx]+1)
	while convergence < R:
		for k in range(10): 
			i = 0
			for x_curr in x_new:
				y_predict = np.dot(x_curr,weight)
				if y[i] * y_predict <= 0:
					convergence = 0
					error += 1
				else: 
					convergence += 1
					if convergence >= R:
						break
				if y[i] * y_predict <= 1:
					gti = np.multiply(-y[i], x_curr)
					gt += np.multiply(gti, gti)
					for j in range(weight.size):
						if gt[j] != 0:
							weight[j] += r*y[i]*x_curr[j]/math.sqrt(gt[j])
				# mistakes.append(error)
				i += 1
			if convergence >= R:
				break
	return error

#####################################################
l = 10
m = 20
n = [40, 80, 120, 160, 200] # manually change from n=40 to n=200 in increments of 40
margin_perceptron = [0.25, 0.03, 0.03, 0.25, 0.03]
R = 1000
mistakes = []
for i in range(5):
	mistakes.append([])
for idx in range(5):
	(y,x) = gen(l, m, n[idx], 50000, False)

	mistakes[0].append(Perceptron(x, y, 1))
	mistakes[1].append(Perceptron_with_margin(x, y, margin_perceptron[idx], 1))
	mistakes[2].append(Winnow(x, y, 1.1))
	mistakes[3].append(Winnow_with_margin(x, y, 1.1, 2.0))
	mistakes[4].append(AdaGrad(x, y, 1.5))

plt.plot(n, mistakes[0], label = 'Perceptron')
plt.plot(n, mistakes[1], label = 'Perceptron w/ margin')
plt.plot(n, mistakes[2], label = 'Winnow')
plt.plot(n, mistakes[3], label = 'Winnow w/ margin')
plt.plot(n, mistakes[4], label = 'AdaGrad')
plt.title('Experiment 2')
plt.xlabel('n')
plt.ylabel('W')
plt.legend(loc = 2)
plt.show()

