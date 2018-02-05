from __future__ import print_function

import arff
import numpy as np

def training(train_data, train_label, weight, epsilon, r):
	loss = []
	error = 1
	while error >= epsilon:
		for i in range(len(train_data)):
			x = np.array(train_data[i])
			predict = np.inner(weight, x)

			loss.append(np.sum((predict-train_label)**2)/len(train_label))
			if len(loss) == 1:
				error = loss[-1]
			else:
				error = abs(loss[-1]-loss[-2])
			if error < epsilon:
				break

			delta = r*np.multiply((train_label[i]-predict),x)
			weight += delta

	return weight

def evaluate(weight, test_data, test_label):
	accurate = 0
	for i in range(len(test_label)):
		if (np.inner(test_data[i],weight) * test_label[i] >= 0):
			accurate += 1

	return float(accurate)/float(len(test_label))

badges = []
badges.append(list(arff.load("badges.fold1.arff")))
badges.append(list(arff.load("badges.fold2.arff")))
badges.append(list(arff.load("badges.fold3.arff")))
badges.append(list(arff.load("badges.fold4.arff")))
badges.append(list(arff.load("badges.fold5.arff")))

data = []
label = []
for i in range(5):
	data.append([])
	label.append([])
	for j in range(len(badges[i])):
		data[-1].append([])
		for k in range(260):
			data[-1][-1].append(float(badges[i][j][k]))
		data[-1][-1].append(float(-1)) # move to (n+1) dimensional representation (x, -1)
		if badges[i][j][k+1] == '+':
			label[-1].append(1)
		elif badges[i][j][k+1] == '-':
			label[-1].append(-1)

weight = np.zeros(261) # (n+1) dimensions

epsilon = 0.0001
r = 0.1
accuracy = []
std = []
for i in range(5):
	test_data = data[i]
	test_label = label[i]
	train_data = []
	train_label = []
	for j in range(5):
		if j!=i:
			train_data += data[j]
			train_label += label[j]

	weight = training(train_data, train_label, weight, epsilon, r)

	accuracy.append(evaluate(weight, test_data, test_label))

print(accuracy)
print(np.mean(accuracy))
print(np.std(accuracy))


