from __future__ import print_function
import NN, data_loader, perceptron, random
import numpy as np

# training_data, test_data = data_loader.load_circle_data()
training_data, test_data = data_loader.load_mnist_data()

num = len(training_data)
random.shuffle(training_data)
fold_train = []
fold_test = []
for i in range(5):
	fold_train.append(training_data[num/5*i : num/5*(i+1)])
	fold_test.append( training_data[0 : num/5*i] + training_data[num/5*(i+1) : num] )

batch_size = [10, 50, 100]
learning_rate = [0.01, 0.1]
activation_function = ['tanh', 'relu']
hidden_layer_width = [10, 50]


# domain = 'circles'
domain = 'mnist'
data_dim = len(training_data[0][0])
print('dataset: mnist')
for bs in batch_size:
	for rate in learning_rate:
		for func in activation_function:
			for width in hidden_layer_width:
				accuracy = []
				for i in range(5):
					net = NN.create_NN(domain, bs, rate, func, width)
					net.train(fold_train[i])
					accuracy.append(net.evaluate(fold_test[i]))
				print('Batch Size: '+str(bs)+' Learning Rate: '+str(rate)+' Activation Function: '+func+' Hidden Layer Width: '+str(width)+ '  Accuracy: ' +str(np.mean(accuracy)))




