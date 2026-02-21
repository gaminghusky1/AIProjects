import mlx.core as mx
import math
from mlx_model import activations

def sliding_window_view(x, window_shape, axis):
    """
    MLX batch-safe equivalent of numpy.lib.stride_tricks.sliding_window_view
    for x shaped (B, C, H, W) with axis=(2,3) and window_shape=(Kh,Kw).

    Returns shape: (B, C, H_out, W_out, Kh, Kw)
    """
    if tuple(axis) != (2, 3):
        raise NotImplementedError("This implementation supports axis=(2,3) only.")
    if len(window_shape) != 2:
        raise ValueError("window_shape must be (Kh, Kw).")

    Kh, Kw = window_shape
    B, C, H, W = x.shape

    H_out = H - Kh + 1
    W_out = W - Kw + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"window_shape {window_shape} too large for input {(H, W)}")

    # Indices for sliding windows
    ih = mx.arange(H_out)[:, None] + mx.arange(Kh)[None, :]   # (H_out, Kh)
    iw = mx.arange(W_out)[:, None] + mx.arange(Kw)[None, :]   # (W_out, Kw)

    # 1) Gather along H (axis=2)
    ih_flat = mx.reshape(ih, (-1,))                           # (H_out*Kh,)
    xh = mx.take(x, ih_flat, axis=2)                          # (B, C, H_out*Kh, W)
    xh = mx.reshape(xh, (B, C, H_out, Kh, W))                 # (B, C, H_out, Kh, W)

    # 2) Gather along W (now axis=4)
    iw_flat = mx.reshape(iw, (-1,))                           # (W_out*Kw,)
    xhw = mx.take(xh, iw_flat, axis=4)                        # (B, C, H_out, Kh, W_out*Kw)
    xhw = mx.reshape(xhw, (B, C, H_out, Kh, W_out, Kw))       # (B, C, H_out, Kh, W_out, Kw)

    # Reorder to (B, C, H_out, W_out, Kh, Kw) like NumPy
    windows = mx.transpose(xhw, (0, 1, 2, 4, 3, 5))
    return windows

class Dense:
    def __init__(self, units, activation='linear'):
        self.units = units
        self.activation_name = activation
        self.activation = activations.Activation(activation)
        self.weights = None
        self.weights_gradient = None
        self.biases = None
        self.biases_gradient = None
        self.input_shape = None

    def init_weights(self, last_layer_shape):
        self.input_shape = last_layer_shape
        in_features = int(math.prod(last_layer_shape))
        self.weights = mx.random.normal((in_features, self.units)) * math.sqrt(2 / in_features)
        self.biases = mx.zeros(self.units)

        self.weights_gradient = mx.zeros_like(self.weights)
        self.biases_gradient = mx.zeros_like(self.biases)

        mx.eval(self.weights, self.biases, self.weights_gradient, self.biases_gradient)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        prev_layer_activations = mx.reshape(prev_layer_activations, (batch_size, -1))
        z_output = mx.matmul(prev_layer_activations, self.weights) + self.biases
        a_output = self.activation(z_output)

        return a_output, z_output

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        prev_layer_activations = mx.reshape(prev_layer_activations, (batch_size, -1))
        curr_layer_z = mx.reshape(curr_layer_z, (batch_size, -1))
        dc_da = mx.reshape(dc_da, (batch_size, -1))
        if self.activation.elementwise:
            dc_dz = dc_da * self.activation.derivative(curr_layer_z)
        else:
            dc_dz = mx.einsum('bo,boo->bo', dc_da, self.activation.derivative(curr_layer_z))
        # Cost of current layer weights
        # dc_dw; dz_dw = a_L-1
        self.weights_gradient = self.weights_gradient + mx.matmul(prev_layer_activations.T, dc_dz)
        # Cost of biases
        # dz_db = 1 because biases are constants
        self.biases_gradient = self.biases_gradient + mx.sum(dc_dz, axis=0)

        # Cost of previous layer activations
        # dc_d(a_L-1); dz_da-1 = self.weights
        return mx.matmul(dc_dz, self.weights.T)

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.weights = self.weights - learning_rate * self.weights_gradient
        self.biases = self.biases - learning_rate * self.biases_gradient

        mx.eval(self.weights, self.biases)

    def reset_grads(self):
        self.weights_gradient = mx.zeros_like(self.weights)
        self.biases_gradient = mx.zeros_like(self.biases)

        mx.eval(self.weights_gradient, self.biases_gradient)

    def get_param_refs(self):
        return [(self, "weights"), (self, "biases")]

    def get_grads(self):
        return [self.weights_gradient, self.biases_gradient]

    def get_param_count(self):
        return math.prod(self.weights.shape) + math.prod(self.biases.shape)

    def get_output_shape(self):
        return (self.units,)

class Convolution:
    def __init__(self, num_filters, kernel_shape, activation='linear', input_shape=None, stride=1, padding=0):
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.input_size = None
        self.kernel_shape = kernel_shape
        self.kernel_size = math.prod(kernel_shape)
        self.kernel_stack_shape = None
        self.kernel_stack_size = None
        self.weights = None
        self.weights_gradient = None
        self.biases = None
        self.biases_gradient = None
        self.activation_name = activation
        self.activation = activations.Activation(activation)
        self.stride = stride
        self.padding = padding
        self.output_shape = None
        self.output_size = None
        self.stacked_output_shape = None
        self.dz_da = None

    def precompute_dz_da(self):
        self.dz_da = mx.zeros((*self.stacked_output_shape, *self.input_shape))
        for f in range(self.num_filters):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    i_start = i * self.stride
                    j_start = j * self.stride
                    self.dz_da[f, i, j, :, i_start:i_start + self.kernel_shape[0], j_start:j_start + self.kernel_shape[1]] = self.weights[f]
        self.dz_da = mx.reshape(self.dz_da, (self.num_filters, self.output_size, -1))

    def init_weights(self, last_layer_shape):
        if self.input_shape is None:
            self.input_shape = last_layer_shape
        if len(self.input_shape) == 2:
            self.input_shape = (1, *self.input_shape)
        self.input_size = math.prod(self.input_shape)
        self.kernel_stack_shape = (self.input_shape[0], *self.kernel_shape)
        self.kernel_stack_size = math.prod(self.kernel_stack_shape)
        self.output_shape = ((self.input_shape[1] - self.kernel_shape[0] + 2 * self.padding) // self.stride + 1, (self.input_shape[2] - self.kernel_shape[1] + 2 * self.padding) // self.stride + 1)
        self.output_size = math.prod(self.output_shape)
        self.stacked_output_shape = (self.num_filters, *self.output_shape)
        self.weights = mx.random.normal((self.num_filters, *self.kernel_stack_shape)) * math.sqrt(2 / self.kernel_stack_size)
        self.weights_gradient = mx.zeros(self.weights.shape)
        self.biases = mx.zeros(self.num_filters)
        self.biases_gradient = mx.zeros(self.biases.shape)

        self.precompute_dz_da()
        mx.eval(self.weights, self.biases, self.weights_gradient, self.biases_gradient, self.dz_da)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def get_windows(self, prev_layer_activations, batch_size):
        prev_layer_activations = mx.reshape(prev_layer_activations, (batch_size, *self.input_shape))
        prev_layer_activations = mx.pad(prev_layer_activations, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant', constant_values=0)
        windows = sliding_window_view(prev_layer_activations, window_shape=self.kernel_shape, axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :].transpose((0, 2, 3, 1, 4, 5))
        windows = mx.reshape(windows, (batch_size, self.output_size, self.kernel_stack_size))
        return windows

    def forward_pass(self, prev_layer_activations, batch_size):
        windows = self.get_windows(prev_layer_activations, batch_size)
        flattened_weights = mx.reshape(self.weights, (self.num_filters, self.kernel_stack_size))
        z_output = mx.einsum('bok,fk->bfo', windows, flattened_weights)
        z_output = mx.reshape(z_output, (batch_size, self.num_filters, *self.output_shape))
        z_output = z_output + self.biases[:, mx.newaxis, mx.newaxis]
        # for f in range(self.num_filters):
        #     weight_matrix = self.weights[f]
        #     bias = self.biases[f]
        #     for i in range(self.output_shape[0]):
        #         for j in range(self.output_shape[1]):
        #             i_start = i * self.stride
        #             j_start = j * self.stride
        #             prev_layer_activations_window = prev_layer_activations[:, i_start:i_start + self.kernel_shape[0], j_start:j_start + self.kernel_shape[1]]
        #             z_output[f, i, j] = np.sum(prev_layer_activations_window * weight_matrix) + bias
        a_output = self.activation(z_output)
        return a_output, z_output

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        curr_layer_z = mx.reshape(curr_layer_z, (batch_size, self.num_filters, self.output_size))
        dc_da = mx.reshape(dc_da, (batch_size, self.num_filters, self.output_size))
        da_dz = self.activation.derivative(curr_layer_z)
        if self.activation.elementwise:
            dc_dz = dc_da * da_dz
        else:
            dc_dz = mx.einsum('bfo,bfoo->bfo', dc_da, da_dz)

        windows = self.get_windows(prev_layer_activations, batch_size)
        # dz_dw = np.zeros((*self.stacked_output_shape, *self.kernel_stack_shape))
        # for f in range(self.num_filters):
        #     for i in range(self.output_shape[0]):
        #         for j in range(self.output_shape[1]):
        #             i_start = i * self.stride
        #             j_start = j * self.stride
        #             dz_dw[f, i, j, :, :, :] = prev_layer_activations[:, i_start:i_start + self.kernel_shape[0], j_start:j_start + self.kernel_shape[1]]
        # dz_dw = np.transpose(dz_dw, (0, 3, 4, 5, 1, 2))
        # dz_dw = np.reshape(dz_dw, (self.num_filters, self.kernel_stack_size, -1))

        dc_dw = mx.einsum('bfo,bok->fk', dc_dz, windows)
        self.weights_gradient = self.weights_gradient + mx.reshape(dc_dw, (self.num_filters, *self.kernel_stack_shape))
        self.biases_gradient = self.biases_gradient + mx.sum(dc_dz, axis=(0, 2))
        dc_da = mx.einsum('bfo,foi->bi', dc_dz, self.dz_da)
        return dc_da

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.weights = self.weights - learning_rate * self.weights_gradient
        self.biases = self.biases - learning_rate * self.biases_gradient

        mx.eval(self.weights, self.biases)

    def reset_grads(self):
        self.weights_gradient = mx.zeros_like(self.weights)
        self.biases_gradient = mx.zeros_like(self.biases)

        self.precompute_dz_da()
        mx.eval(self.weights_gradient, self.biases_gradient, self.dz_da)

    def get_param_refs(self):
        return [(self, "weights"), (self, "biases")]

    def get_grads(self):
        return [self.weights_gradient, self.biases_gradient]

    def get_param_count(self):
        return math.prod(self.weights.shape) + math.prod(self.biases.shape)

    def get_output_shape(self):
        return self.stacked_output_shape

class MaxPooling:
    def __init__(self, pool_size, stride=1, padding=0, input_shape=None):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.output_shape = None
        self.output_size = None

    def init_weights(self, last_layer_shape):
        if self.input_shape is None:
            self.input_shape = last_layer_shape
        if len(self.input_shape) == 2:
            self.input_shape = (1, *self.input_shape)
        self.output_shape = (self.input_shape[0], (self.input_shape[1] - self.pool_size[0] + 2 * self.padding) // self.stride + 1, (self.input_shape[2] - self.pool_size[1] + 2 * self.padding) // self.stride + 1)
        self.output_size = math.prod(self.output_shape)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        prev_layer_activations = mx.reshape(prev_layer_activations, (batch_size, *self.input_shape))
        prev_layer_activations = mx.pad(prev_layer_activations, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant', constant_values=0)
        # a_output = np.zeros(self.output_shape)
        windows = sliding_window_view(prev_layer_activations, window_shape=self.pool_size, axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        a_output = mx.max(windows, axis=(4, 5))
        # max_indices = np.zeros(self.output_shape, dtype=int)
        # for c in range(self.output_shape[0]):
        #     for i in range(self.output_shape[1]):
        #         for j in range(self.output_shape[2]):
        #             i_start = i * self.stride
        #             j_start = j * self.stride
        #             prev_layer_activations_window = prev_layer_activations[c, i_start:i_start + self.pool_size[0], j_start:j_start + self.pool_size[1]]
        #             a_output[c, i, j] = np.max(prev_layer_activations_window)
        #             max_indices[c, i, j] = np.argmax(prev_layer_activations_window)
        return a_output, windows

    def backward_pass(self, prev_layer_activations, windows, dc_da, batch_size):
        prev_layer_activations = mx.reshape(prev_layer_activations, (batch_size, *self.input_shape))
        # dc_da = dc_da.reshape(self.output_shape)
        # prev_layer_activations = np.pad(prev_layer_activations, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        # prev_dc_da = np.zeros(self.input_shape)
        # for c in range(self.output_shape[0]):
        #     for i in range(self.output_shape[1]):
        #         for j in range(self.output_shape[2]):
        #             i_start = i * self.stride
        #             j_start = j * self.stride
        #             prev_layer_activations_window = prev_layer_activations[c, i_start:i_start + self.pool_size[0], j_start:j_start + self.pool_size[1]]
        #             idx = np.unravel_index(max_indices[c, i, j], prev_layer_activations_window.shape)
        #             prev_dc_da[c, i_start + idx[0], j_start + idx[1]] += dc_da[c, i, j]
        max_areas = (windows == mx.max(windows, axis=(4, 5), keepdims=True))
        dc_da = mx.reshape(dc_da, (batch_size, *self.output_shape))
        max_areas_scaled = max_areas * dc_da[..., mx.newaxis, mx.newaxis]
        prev_dc_da = mx.zeros_like(prev_layer_activations)
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                i_start = i * self.stride
                j_start = j * self.stride
                prev_dc_da[:, :, i_start:i_start + self.pool_size[0], j_start:j_start + self.pool_size[1]] += max_areas_scaled[:, :, i, j, :, :]
        return prev_dc_da

    def update_weights_and_biases(self, learning_rate, batch_size):
        pass

    def reset_grads(self):
        pass

    def get_param_refs(self):
        return []

    def get_grads(self):
        return []

    def get_param_count(self):
        return 0

    def get_output_shape(self):
        return self.output_shape

class TokenEmbedding:
    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weights = None
        self.weights_gradient = None
        self.sequence_len = None

    def init_weights(self, last_layer_shape):
        self.sequence_len = last_layer_shape
        self.weights = mx.random.normal((self.vocab_size, self.embedding_size)) / math.sqrt(self.embedding_size)
        self.weights_gradient = mx.zeros_like(self.weights)

        mx.eval(self.weights, self.weights_gradient)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        prev_layer_activations = mx.reshape(prev_layer_activations, (batch_size, -1))
        a_output = self.weights[prev_layer_activations]

        return a_output, None

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        dc_da = mx.reshape(dc_da, (-1, self.embedding_size))

        self.weights_gradient.at[mx.flatten(prev_layer_activations)].add(dc_da)

        return None

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.weights = self.weights - learning_rate * self.weights_gradient

        mx.eval(self.weights)

    def reset_grads(self):
        self.weights_gradient = mx.zeros_like(self.weights)

        mx.eval(self.weights_gradient)

    def get_param_refs(self):
        return [(self, "weights")]

    def get_grads(self):
        return [self.weights_gradient]

    def get_param_count(self):
        return math.prod(self.weights.shape)

    def get_output_shape(self):
        return (*self.sequence_len, self.embedding_size)

class PositionalEmbedding:
    def __init__(self, max_sequence_len=512):
        self.max_sequence_len = max_sequence_len
        self.weights = None
        self.weights_gradient = None
        self.input_shape = None
        self.embedding_size = None

    def init_weights(self, last_layer_shape):
        self.input_shape = last_layer_shape
        self.embedding_size = self.input_shape[1]
        self.weights = mx.random.normal((self.max_sequence_len, self.embedding_size)) / math.sqrt(self.embedding_size)
        self.weights_gradient = mx.zeros_like(self.weights)

        mx.eval(self.weights, self.weights_gradient)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        sequence_len = prev_layer_activations.shape[1]
        a_output = prev_layer_activations + self.weights[:sequence_len]

        return a_output, None

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        sequence_len = prev_layer_activations.shape[1]
        self.weights_gradient[:sequence_len] += mx.sum(dc_da, axis=0)

        return dc_da

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.weights = self.weights - learning_rate * self.weights_gradient

        mx.eval(self.weights)

    def reset_grads(self):
        self.weights_gradient = mx.zeros_like(self.weights)

        mx.eval(self.weights_gradient)

    def get_param_refs(self):
        return [(self, "weights")]

    def get_grads(self):
        return [self.weights_gradient]

    def get_param_count(self):
        return math.prod(self.weights.shape)

    def get_output_shape(self):
        return self.input_shape

class LayerNorm:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.scale = None
        self.scale_gradient = None
        self.shift = None
        self.shift_gradient = None
        self.input_shape = None
        self.sequence_len = None

    def init_weights(self, last_layer_shape):
        self.input_shape = last_layer_shape
        self.sequence_len = self.input_shape[0]
        self.scale = mx.ones(self.input_shape[1])
        self.shift = mx.zeros(self.input_shape[1])

        self.scale_gradient = mx.zeros_like(self.scale)
        self.shift_gradient = mx.zeros_like(self.shift)

        mx.eval(self.scale, self.shift, self.scale_gradient, self.shift_gradient)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        numerator_term = prev_layer_activations - mx.mean(prev_layer_activations, axis=2, keepdims=True)
        denominator_term = mx.sqrt(mx.var(prev_layer_activations, axis=2, keepdims=True) + self.epsilon)

        x_hat = numerator_term / denominator_term

        a_output = self.scale * x_hat + self.shift

        return a_output, (x_hat, denominator_term)

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        x_hat, denominator_term = curr_layer_z

        self.scale_gradient = self.scale_gradient + mx.sum(dc_da * x_hat, axis=(0, 1))
        self.shift_gradient = mx.sum(dc_da, axis=(0, 1))

        g = dc_da * self.scale

        dc_dprev_layer_activations = (g - g.mean(axis=2, keepdims=True) - x_hat * mx.mean(g * x_hat, axis=2, keepdims=True)) / denominator_term
        return dc_dprev_layer_activations

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.scale = self.scale - learning_rate * self.scale_gradient
        self.shift = self.shift - learning_rate * self.shift_gradient

        mx.eval(self.scale, self.shift)

    def reset_grads(self):
        self.scale_gradient = mx.zeros_like(self.scale)
        self.shift_gradient = mx.zeros_like(self.shift)

        mx.eval(self.scale_gradient, self.shift_gradient)

    def get_param_refs(self):
        return [(self, "scale"), (self, "shift")]

    def get_grads(self):
        return [self.scale_gradient, self.shift_gradient]

    def get_param_count(self):
        return math.prod(self.scale.shape) + math.prod(self.shift.shape)

    def get_output_shape(self):
        return self.input_shape

class Attention:
    def __init__(self, d_k, d_v, heads=1, mask=None):
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads

        self.d_i = None

        self.w_q = None
        self.w_k = None
        self.w_v = None
        self.w_o = None
        self.b_o = None

        self.w_q_gradient = None
        self.w_k_gradient = None
        self.w_v_gradient = None
        self.w_o_gradient = None
        self.b_o_gradient = None

        self.mask = mask

        self.input_shape = None

    def init_weights(self, last_layer_shape):
        self.input_shape = last_layer_shape
        self.d_i = self.input_shape[1]

        self.w_q = mx.random.normal((self.d_i, self.d_k)) * math.sqrt(2 / self.d_i)
        self.w_k = mx.random.normal((self.d_i, self.d_k)) * math.sqrt(2 / self.d_i)
        self.w_v = mx.random.normal((self.d_i, self.d_v)) * math.sqrt(2 / self.d_i)
        self.w_o = mx.random.normal((self.d_v, self.d_i)) * math.sqrt(2 / self.d_v)
        self.b_o = mx.zeros(self.d_i)

        self.w_q_gradient = mx.zeros_like(self.w_q)
        self.w_k_gradient = mx.zeros_like(self.w_k)
        self.w_v_gradient = mx.zeros_like(self.w_v)
        self.w_o_gradient = mx.zeros_like(self.w_o)
        self.b_o_gradient = mx.zeros_like(self.b_o)

        mx.eval(self.w_q, self.w_k, self.w_v, self.w_o, self.b_o, self.w_q_gradient, self.w_k_gradient, self.w_v_gradient, self.w_o_gradient, self.b_o_gradient)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        sequence_len = prev_layer_activations.shape[1]
        # t and s are both sequence_length, this is just to distinguish the q dimension and the k dimension for einsum.
        q = mx.einsum('ik,bti->btk', self.w_q, prev_layer_activations)
        k = mx.einsum('ik,bsi->bsk', self.w_k, prev_layer_activations)
        v = mx.einsum('iv,bsi->bsv', self.w_v, prev_layer_activations)
        # print("q before:", q.shape)

        # split each into heads for multihead attention
        q = mx.reshape(q, (batch_size, sequence_len, self.heads, -1)).transpose((0, 2, 1, 3))
        # print("q after:", q.shape)
        k = mx.reshape(k, (batch_size, sequence_len, self.heads, -1)).transpose((0, 2, 1, 3))
        v = mx.reshape(v, (batch_size, sequence_len, self.heads, -1)).transpose((0, 2, 1, 3))

        raw_scores = mx.einsum('bhtk,bhsk->bhts', q, k) / math.sqrt(self.d_k // self.heads)
        if self.mask is not None:
            raw_scores = mx.where(self.mask(sequence_len), -1e9, raw_scores)

        attention_scores = activations.softmax(raw_scores)

        z_output = mx.einsum('bhts,bhsv->bhtv', attention_scores, v)
        z_output = z_output.transpose((0, 2, 1, 3)).reshape((batch_size, sequence_len, -1))

        # print("z_output shape:", z_output.shape)
        # print("w_o shape:", self.w_o.shape)
        a_output = mx.einsum('vi,btv->bti', self.w_o, z_output)
        a_output = a_output + self.b_o

        return a_output, (raw_scores, attention_scores, q, k, v, z_output)

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        raw_scores, attention_scores, q, k, v, z_output = curr_layer_z
        sequence_len = prev_layer_activations.shape[1]

        self.w_o_gradient = self.w_o_gradient + mx.einsum('bsi,bsv->vi', dc_da, z_output)
        self.b_o_gradient = self.b_o_gradient + mx.sum(dc_da, axis=(0, 1))

        dc_dz = mx.einsum('bsi,vi->bsv', dc_da, self.w_o)
        dc_dz = mx.reshape(dc_dz, (batch_size, sequence_len, self.heads, -1)).transpose((0, 2, 1, 3))

        dc_dv = mx.einsum('bhtv,bhts->bhsv', dc_dz, attention_scores)
        dc_dv = mx.transpose(dc_dv, (0, 2, 1, 3)).reshape((batch_size, sequence_len, -1))
        self.w_v_gradient = self.w_v_gradient + mx.einsum('bsv,bsi->iv', dc_dv, prev_layer_activations)

        dc_dattention_scores = mx.einsum('bhtv,bhsv->bhts', dc_dz, v)
        dattention_scores_draw_scores = activations.attention_softmax_derivative(raw_scores)

        dc_draw_scores = mx.einsum('bhts,bhtss->bhts', dc_dattention_scores, dattention_scores_draw_scores)
        if self.mask is not None:
            dc_draw_scores = mx.where(self.mask(sequence_len), 0, dc_draw_scores)

        dc_dq = mx.einsum('bhts,bhsk->bhtk', dc_draw_scores, k) / math.sqrt(self.d_k // self.heads)
        dc_dk = mx.einsum('bhts,bhtk->bhsk', dc_draw_scores, q) / math.sqrt(self.d_k // self.heads)

        dc_dq = mx.transpose(dc_dq, (0, 2, 1, 3)).reshape((batch_size, sequence_len, -1))
        dc_dk = mx.transpose(dc_dk, (0, 2, 1, 3)).reshape((batch_size, sequence_len, -1))

        self.w_q_gradient = self.w_q_gradient + mx.einsum('btk,bti->ik', dc_dq, prev_layer_activations)
        self.w_k_gradient = self.w_k_gradient + mx.einsum('bsk,bsi->ik', dc_dk, prev_layer_activations)

        dc_dprev_layer_activations = mx.einsum('btk,ik->bti', dc_dq, self.w_q) + mx.einsum('bsk,ik->bsi', dc_dk, self.w_k) + mx.einsum('bsv,iv->bsi', dc_dv, self.w_v)
        return dc_dprev_layer_activations


    def update_weights_and_biases(self, learning_rate, batch_size):
        self.w_q = self.w_q - learning_rate * self.w_q_gradient
        self.w_k = self.w_k - learning_rate * self.w_k_gradient
        self.w_v = self.w_v - learning_rate * self.w_v_gradient
        self.w_o = self.w_o - learning_rate * self.w_o_gradient
        self.b_o = self.b_o - learning_rate * self.b_o_gradient

        mx.eval(self.w_q, self.w_k, self.w_v, self.w_o, self.b_o)

    def reset_grads(self):
        self.w_q_gradient = mx.zeros_like(self.w_q)
        self.w_k_gradient = mx.zeros_like(self.w_k)
        self.w_v_gradient = mx.zeros_like(self.w_v)
        self.w_o_gradient = mx.zeros_like(self.w_o)
        self.b_o_gradient = mx.zeros_like(self.b_o)

    def get_param_refs(self):
        return [(self, "w_q"), (self, "w_k"), (self, "w_v"), (self, "w_o"), (self, "b_o")]

    def get_grads(self):
        return [self.w_q_gradient, self.w_k_gradient, self.w_v_gradient, self.w_o_gradient, self.b_o_gradient]

    def get_param_count(self):
        return math.prod(self.w_q.shape) + math.prod(self.w_k.shape) + math.prod(self.w_v.shape) + math.prod(self.w_o.shape) + math.prod(self.b_o.shape)

    def get_output_shape(self):
        return self.input_shape

class MultilayerPerceptron:
    def __init__(self, d_ff, activation='gelu'):
        self.d_ff = d_ff
        self.activation_name = activation
        self.activation = activations.Activation(activation)

        self.up_weights = None
        self.up_weights_gradient = None
        self.up_biases = None
        self.up_biases_gradient = None

        self.down_weights = None
        self.down_weights_gradient = None
        self.down_biases = None
        self.down_biases_gradient = None

        self.input_shape = None
        self.d_i = None

    def init_weights(self, last_layer_shape):
        self.input_shape = last_layer_shape
        self.d_i = self.input_shape[1]

        self.up_weights = mx.random.normal((self.d_ff, self.d_i)) * math.sqrt(2 / self.d_i)
        self.up_weights_gradient = mx.zeros_like(self.up_weights)
        self.up_biases = mx.zeros(self.d_ff)
        self.up_biases_gradient = mx.zeros_like(self.up_biases)

        self.down_weights = mx.random.normal((self.d_i, self.d_ff)) * math.sqrt(2 / self.d_ff)
        self.down_weights_gradient = mx.zeros_like(self.down_weights)
        self.down_biases = mx.zeros(self.d_i)
        self.down_biases_gradient = mx.zeros_like(self.down_biases)

        mx.eval(self.up_weights, self.up_biases, self.down_weights, self.down_biases, self.up_weights_gradient, self.up_biases_gradient, self.down_weights_gradient, self.down_biases_gradient)

    def get_ouptut(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        raw_ff = mx.einsum('fi,bsi->bsf', self.up_weights, prev_layer_activations)
        raw_ff = raw_ff + self.up_biases

        activated_ff = self.activation(raw_ff)

        a_output = mx.einsum('if,bsf->bsi', self.down_weights, activated_ff)
        a_output = a_output + self.down_biases

        return a_output, (raw_ff, activated_ff)

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        raw_ff, activated_ff = curr_layer_z

        self.down_weights_gradient = self.down_weights_gradient + mx.einsum('bsi,bsf->if', dc_da, activated_ff)
        self.down_biases_gradient = self.down_biases_gradient + mx.sum(dc_da, axis=(0, 1))

        dc_dactivated_ff = mx.einsum('if,bsi->bsf', self.down_weights, dc_da)
        dc_draw_ff = dc_dactivated_ff * self.activation.derivative(raw_ff)

        self.up_weights_gradient = self.up_weights_gradient + mx.einsum('bsf,bsi->fi', dc_draw_ff, prev_layer_activations)
        self.up_biases_gradient = self.up_biases_gradient + mx.sum(dc_draw_ff, axis=(0, 1))

        # dc_da += np.einsum('fi,bsf->bsi', self.up_weights, dc_draw_ff)
        dc_dprev_layer_activations = mx.einsum('fi,bsf->bsi', self.up_weights, dc_draw_ff)
        return dc_dprev_layer_activations

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.up_weights = self.up_weights - learning_rate * self.up_weights_gradient
        self.up_biases = self.up_biases - learning_rate * self.up_biases_gradient

        self.down_weights = self.down_weights - learning_rate * self.down_weights_gradient
        self.down_biases = self.down_biases - learning_rate * self.down_biases_gradient

        mx.eval(self.up_weights, self.up_biases, self.down_weights, self.down_biases)

    def reset_grads(self):
        self.up_weights_gradient = mx.zeros_like(self.up_weights)
        self.up_biases_gradient = mx.zeros_like(self.up_biases)
        self.down_weights_gradient = mx.zeros_like(self.down_weights)
        self.down_biases_gradient = mx.zeros_like(self.down_biases)

        mx.eval(self.up_weights_gradient, self.up_biases_gradient, self.down_weights_gradient, self.down_biases_gradient)

    def get_param_refs(self):
        return [(self, "up_weights"), (self, "up_biases"), (self, "down_weights"), (self, "down_biases")]

    def get_grads(self):
        return [self.up_weights_gradient, self.up_biases_gradient, self.down_weights_gradient, self.down_biases_gradient]

    def get_param_count(self):
        return math.prod(self.up_weights.shape) + math.prod(self.up_biases.shape) + math.prod(self.down_weights.shape) + math.prod(self.down_biases.shape)

    def get_output_shape(self):
        return self.input_shape

class ResidualBlock:
    def __init__(self, *layers):
        self.layers = layers
        self.output_shape = None

    def init_weights(self, last_layer_shape):
        for layer in self.layers:
            layer.init_weights(last_layer_shape)
            last_layer_shape = layer.get_output_shape()
        self.output_shape = last_layer_shape

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_outputs = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        a_outputs = [prev_layer_activations]
        z_outputs = []
        for layer in self.layers:
            a_output, z_output = layer.forward_pass(a_outputs[-1], batch_size)
            a_outputs.append(a_output)
            z_outputs.append(z_output)
        a_outputs[-1] += prev_layer_activations
        return a_outputs[-1], (a_outputs, z_outputs)

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        a_outputs, z_outputs = curr_layer_z
        dc_da_curr = dc_da
        for i in reversed(range(len(self.layers))):
            dc_da_curr = self.layers[i].backward_pass(a_outputs[i], z_outputs[i], dc_da_curr, batch_size)
        return dc_da_curr + dc_da

    def update_weights_and_biases(self, learning_rate, batch_size):
        for layer in self.layers:
            layer.update_weights_and_biases(learning_rate, batch_size)

    def reset_grads(self):
        for layer in self.layers:
            layer.reset_grads()

    def get_param_refs(self):
        param_refs = []
        for layer in self.layers:
            param_refs.extend(layer.get_param_refs())
        return param_refs

    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_grads())
        return grads

    def get_param_count(self):
        param_count = 0
        for layer in self.layers:
            param_count += layer.get_param_count()
        return param_count

    def get_output_shape(self):
        return self.output_shape

class TimeDistributedDense:
    def __init__(self, units, activation='linear'):
        self.activation_name = activation
        self.activation = activations.Activation(activation)

        self.units = units

        self.weights = None
        self.weights_gradient = None
        self.biases = None
        self.biases_gradient = None

        self.output_shape = None
        self.d_i = None

    def init_weights(self, last_layer_shape):
        self.output_shape = (last_layer_shape[0], self.units)
        self.d_i = last_layer_shape[1]

        self.weights = mx.random.normal((self.units, self.d_i)) * math.sqrt(2 / self.d_i)
        self.weights_gradient = mx.zeros_like(self.weights)
        self.biases = mx.zeros(self.units)
        self.biases_gradient = mx.zeros_like(self.biases)

        mx.eval(self.weights, self.biases, self.weights_gradient, self.biases_gradient)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        z_output = mx.einsum('oi,bsi->bso', self.weights, prev_layer_activations)
        z_output = z_output + self.biases

        a_output = self.activation(z_output)

        return a_output, z_output

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        if self.activation.elementwise:
            dc_dz = dc_da * self.activation.derivative(curr_layer_z)
        else:
            dc_dz = mx.einsum('bso,bsoo->bso', dc_da, self.activation.derivative(curr_layer_z))

        self.weights_gradient = self.weights_gradient + mx.einsum('bso,bsi->oi', dc_dz, prev_layer_activations)
        self.biases_gradient = self.biases_gradient + mx.sum(dc_dz, axis=(0, 1))

        dc_dprev_layer_activations = mx.einsum('oi,bso->bsi', self.weights, dc_dz)
        return dc_dprev_layer_activations

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.weights = self.weights - learning_rate * self.weights_gradient
        self.biases = self.biases - learning_rate * self.biases_gradient

        mx.eval(self.weights, self.biases)

    def reset_grads(self):
        self.weights_gradient = mx.zeros_like(self.weights)
        self.biases_gradient = mx.zeros_like(self.biases)

        mx.eval(self.weights_gradient, self.biases_gradient)

    def get_param_refs(self):
        return [(self, "weights"), (self, "biases")]

    def get_grads(self):
        return [self.weights_gradient, self.biases_gradient]

    def get_param_count(self):
        return math.prod(self.weights.shape) + math.prod(self.biases.shape)

    def get_output_shape(self):
        return self.output_shape