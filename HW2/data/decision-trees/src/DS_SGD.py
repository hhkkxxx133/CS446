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

            loss.append(np.sum((predict-train_label)**2)/len(train_data))
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

train = []
train.append(list(arff.load("new_trains1.arff")))
train.append(list(arff.load("new_trains2.arff")))
train.append(list(arff.load("new_trains3.arff")))
train.append(list(arff.load("new_trains4.arff")))
train.append(list(arff.load("new_trains5.arff")))

test = []
test.append(list(arff.load("new_test1.arff")))
test.append(list(arff.load("new_test2.arff")))
test.append(list(arff.load("new_test3.arff")))
test.append(list(arff.load("new_test4.arff")))
test.append(list(arff.load("new_test5.arff")))

train_data = []
train_label = []
for i in range(5):
    train_data.append([])
    train_label.append([])
    for j in range(len(train[i])):
        train_data[-1].append([])
        for k in range(100):
            train_data[-1][-1].append(float(train[i][j][k]))
        train_data[-1][-1].append(float(-1)) # move to (n+1) dimensional representation (x, -1)
        # print(train[i][j][100])
        if train[i][j][k+1] == b"+":
            train_label[-1].append(1)
        elif train[i][j][k+1] == b"-":
            train_label[-1].append(-1)
    # print(len(train_data[i]))
    # print(len(train_label[i]))


test_data = []
test_label = []
for i in range(5):
    test_data.append([])
    test_label.append([])
    for j in range(len(test[i])):
        test_data[-1].append([])
        for k in range(100):
            test_data[-1][-1].append(float(test[i][j][k]))
        test_data[-1][-1].append(float(-1)) # move to (n+1) dimensional representation (x, -1)
        if test[i][j][k+1] == b"+":
            test_label[-1].append(1)
        elif test[i][j][k+1] == b"-":
            test_label[-1].append(-1)
    # print(len(test_data[i]))
    # print(len(test_label[i]))

weight = np.zeros(101) # (n+1) dimensions

epsilon = 0.0001
r = 0.1
accuracy = []
std = []
for i in range(5):
    weight = training(train_data[i], train_label[i], weight, epsilon, r)

    accuracy.append(evaluate(weight, test_data[i], test_label[i]))

print(accuracy)
print(np.mean(accuracy))
print(np.std(accuracy))