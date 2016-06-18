
#Standard libaries
import random

#Third party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        #Do not set bias for the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        #Creates a x col, y row matrix of random weights
        #the cols represent the outgoing weights
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, batch_size, learning_rate, test_data = None):
        if test_data: 
            n_test = len(test_data)
       
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]

            for batch in batches:
                self.update_batch(batch, learning_rate)
            
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            
            print("Epoch {0} complete".format(j))

    def update_batch(self, batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate/len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zs = []

        #Feed forward through the net, storing all zs and activations
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #Propagate backwards
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0 / (1.0 + (np.exp(-z)))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def run():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = NumpyNetwork([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    print('done!')
