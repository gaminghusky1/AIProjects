import numpy as np

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    th = np.tanh(x)
    return 1 - th ** 2

def softmax(x):
    x_stable = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    batch_size, n = s.shape
    jacobians = np.zeros((batch_size, n, n))
    for i in range(batch_size):
        s_i = s[i].reshape(-1, 1)
        jacobians[i] = np.diagflat(s_i) - np.dot(s_i, s_i.T)
    return jacobians

def transformer_softmax_derivative(x):
    s = softmax(x)
    b, n, m = s.shape
    diag_indices = np.arange(m)
    diags = np.zeros((b, n, m, m), dtype=s.dtype)
    diags[:, :, diag_indices, diag_indices] = s
    return diags - np.einsum('bni,bnj->bnij', s, s)

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