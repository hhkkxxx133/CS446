from __future__ import print_function
import NN, data_loader, perceptron, random
import numpy as np
import matplotlib.pyplot as plt

# training_data, test_data = data_loader.load_circle_data()
training_data, test_data = data_loader.load_mnist_data()

#### circle parameters
# batch_size = 10
# learning_rate = 0.1
# activation_function = 'relu'
# hidden_layer_width = 50
#### mnist parameters
batch_size = 10
learning_rate = 0.1
activation_function = 'tanh'
hidden_layer_width = 10

# domain = 'circles'
domain = 'mnist'
data_dim = len(training_data[0][0])

net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
accuracy_NN = net.train_with_learning_curve(training_data)
print('NN Accuracy: '+str(net.evaluate(test_data)))

perc = perceptron.Perceptron(data_dim)
accuracy_perc = perc.train_with_learning_curve(training_data)
print('Perceptron Accuracy: '+str(perc.evaluate(test_data)))

xNN = [x[0] for x in accuracy_NN]
yNN = [x[1] for x in accuracy_NN]
xperc = [x[0] for x in accuracy_perc]
yperc = [100*x[1] for x in accuracy_perc]

plt.figure(1)
plt.plot(xNN, yNN, xperc, yperc)
plt.xlabel('Training Step')
plt.ylabel('Accuracy')
plt.legend(['Neural Network', 'Perceptron'],loc='best')
plt.axis([0,100,0,100])
plt.title('Learning Curve of Mnist Dataset') # Circle
plt.show()