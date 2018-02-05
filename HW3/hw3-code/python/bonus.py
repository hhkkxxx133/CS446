from __future__ import print_function
import numpy as np 
import math
import matplotlib.pyplot as plt
from gen import gen


def AdaGrad(x, y, r):
	error_list = []
	loss_list = []
	weight = np.zeros(n+1)
	x_new = np.zeros((10000,n+1))
	for row in range(10000):
		x_new[row,:] = np.concatenate((x[row,:], [1]), axis = 0)
	gt = np.zeros(n+1)

	for rounds in range(50):
		i = 0
		error = 0
		loss = 0
		for x_curr in x_new:
			y_predict = np.dot(x_curr,weight)
			if y[i] * y_predict <= 0:
				error += 1
			if y[i] * y_predict <= 1:
				loss += (1-y[i]*np.dot(x_curr, weight))
				gti = np.multiply(-y[i], x_curr)
				gt += np.multiply(gti, gti)
				for j in range(weight.size):
					if gt[j] != 0:
						weight[j] += r*y[i]*x_curr[j]/math.sqrt(gt[j])
			i += 1
		error_list.append(error)
		loss_list.append(loss)
	return error_list,loss_list

##############################################
l = 10
m = 20
n = 40
(data_y, data_x) = gen(l, m, n, 10000, True)

error_list, loss_list = AdaGrad(data_x, data_y, 1.5)
plt.figure(1)
plt.plot(range(1,50),error_list[1:50])
plt.xlabel('Number of training rounds')
plt.ylabel('Misclassification error')

plt.figure(2)
plt.plot(range(1,50),loss_list[1:50])
plt.xlabel('Number of training rounds')
plt.ylabel('Risk(loss over the dataset)')

plt.show()


