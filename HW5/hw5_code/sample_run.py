import NN, data_loader, perceptron

#training_data, test_data = data_loader.load_circle_data()
training_data, test_data = data_loader.load_mnist_data()

#domain = circles
domain = 'mnist'
batch_size = 10 #[10, 50, 100]
learning_rate = 0.1 # [0.01, 0.1]
activation_function = 'tanh' #'relu'
hidden_layer_width = 5 # [10, 50]
data_dim = len(training_data[0][0])


net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
# print net.train(training_data)
# print net.evaluate(test_data)


net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
print net.train_with_learning_curve(training_data)
# print net.evaluate(test_data)

perc = perceptron.Perceptron(data_dim)
# print perc.train(training_data)
# print perc.evaluate(test_data)
print 1111111
perc = perceptron.Perceptron(data_dim)
print perc.train_with_learning_curve(training_data)
# print perc.evaluate(test_data)

