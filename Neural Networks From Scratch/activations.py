import numpy as np

def linear(x):
    return x

def linear_derivative(x):
    return 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(int)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    x_stable = x - np.max(x)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x)

def softmax_derivative(x):
    s = softmax(x)
    s = s.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

activation_dict = {
    'linear': [linear, linear_derivative, True],
    'sigmoid': [sigmoid, sigmoid_derivative, True],
    'relu': [relu, relu_derivative, True],
    'tanh': [tanh, tanh_derivative, True],
    'softmax': [softmax, softmax_derivative, False]
}

class Activation:
    def __init__(self, func_name):
        if func_name not in activation_dict:
            raise ValueError(f'Invalid activation function identifier: "{func_name}"')
        func_list = activation_dict[func_name]
        self.func = func_list[0]
        self.func_deriv = func_list[1]
        self.elementwise = func_list[2]

    def __call__(self, x):
        return self.func(x)

    def derivative(self, x):
        return self.func_deriv(x)