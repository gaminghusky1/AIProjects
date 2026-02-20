import numpy as np

def mse(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))

def mse_derivative(y, y_hat):
    return 2 * (y_hat - y)

def categorical_crossentropy(y, y_hat):
    y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(y_hat)) / (np.prod(y.shape) / y.shape[-1])

def categorical_crossentropy_derivative(y, y_hat):
    y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
    return (-y / y_hat) / (np.prod(y.shape) / y.shape[-1])

def softmax_crossentropy_derivative(y, y_hat):
    return (y_hat - y) / (np.prod(y.shape) / y.shape[-1])

loss_func_dict = {
    'mse': [mse, mse_derivative],
    'categorical_crossentropy': [categorical_crossentropy, categorical_crossentropy_derivative],
    'softmax_crossentropy': [categorical_crossentropy, softmax_crossentropy_derivative],
}

class LossFunction:
    def __init__(self, func_name):
        if func_name not in loss_func_dict:
            raise ValueError(f'Invalid loss function identifier: "{func_name}"')
        func_list = loss_func_dict[func_name]
        self.func = func_list[0]
        self.func_deriv = func_list[1]

    def __call__(self, y, y_hat):
        return self.func(y, y_hat)

    def derivative(self, y, y_hat):
        return self.func_deriv(y, y_hat)