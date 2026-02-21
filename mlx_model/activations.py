import mlx.core as mx
import math

def linear(x):
    return x

def linear_derivative(x):
    return mx.ones_like(x)

def sigmoid(x):
    return 1 / (1 + mx.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return mx.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def gelu(x):
    x = mx.asarray(x)
    a = 0.044715
    k = math.sqrt(2.0 / math.pi)
    u = k * (x + a * x**3)
    return 0.5 * x * (1.0 + mx.tanh(u))

def gelu_derivative(x):
    x = mx.asarray(x)
    a = 0.044715
    k = math.sqrt(2.0 / math.pi)
    u = k * (x + a * x**3)

    t = mx.tanh(u)
    sech2 = 1.0 - t * t
    du_dx = k * (1.0 + 3.0 * a * x * x)

    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * du_dx

def tanh(x):
    return mx.tanh(x)

def tanh_derivative(x):
    th = mx.tanh(x)
    return 1 - th ** 2

def softmax(x):
    x_stable = x - mx.max(x, axis=-1, keepdims=True)
    exp_x = mx.exp(x_stable)
    return exp_x / mx.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    batch_size, n = s.shape
    jacobians = mx.zeros((batch_size, n, n))
    for i in range(batch_size):
        s_i = s[i].reshape(-1, 1)
        jacobians[i] = mx.diag(s_i) - mx.matmul(s_i, s_i.T)
    return jacobians

def time_distributed_softmax_derivative(x):
    s = softmax(x)
    b, n, m = s.shape
    diag_indices = mx.arange(m)
    diags = mx.zeros((b, n, m, m), dtype=s.dtype)
    diags[:, :, diag_indices, diag_indices] = s
    return diags - mx.einsum('bni,bnj->bnij', s, s)

def attention_softmax_derivative(x):
    s = softmax(x)
    b, h, n, m = s.shape
    diag_indices = mx.arange(m)
    diags = mx.zeros((b, h, n, m, m), dtype=s.dtype)
    diags[:, :, :, diag_indices, diag_indices] = s
    return diags - mx.einsum('bhni,bhnj->bhnij', s, s)

activation_dict = {
    'linear': [linear, linear_derivative, True],
    'sigmoid': [sigmoid, sigmoid_derivative, True],
    'relu': [relu, relu_derivative, True],
    'gelu': [gelu, gelu_derivative, True],
    'tanh': [tanh, tanh_derivative, True],
    'softmax': [softmax, softmax_derivative, False],
    'time_distributed_softmax': [softmax, time_distributed_softmax_derivative, False],
    'crossentropy_softmax': [softmax, linear_derivative, True],
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