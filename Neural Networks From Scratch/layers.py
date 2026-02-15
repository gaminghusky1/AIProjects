import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import activations
import sentencepiece as spm

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
        self.weights = np.random.randn(np.prod(last_layer_shape), self.units) * np.sqrt(2 / np.prod(last_layer_shape))
        self.biases = np.zeros(self.units)

        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        prev_layer_activations = prev_layer_activations.reshape((batch_size, -1))
        z_output = np.dot(prev_layer_activations, self.weights) + self.biases
        a_output = self.activation(z_output)

        return a_output, z_output

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        prev_layer_activations = prev_layer_activations.reshape((batch_size, -1))
        curr_layer_z = curr_layer_z.reshape((batch_size, -1))
        dc_da = dc_da.reshape((batch_size, -1))
        if self.activation.elementwise:
            dc_dz = dc_da * self.activation.derivative(curr_layer_z)
        else:
            dc_dz = np.einsum('bo,boo->bo', dc_da, self.activation.derivative(curr_layer_z))
        # Cost of current layer weights
        # dc_dw; dz_dw = a_L-1
        self.weights_gradient += np.dot(prev_layer_activations.T, dc_dz)
        # Cost of biases
        # dz_db = 1 because biases are constants
        self.biases_gradient += np.sum(dc_dz, axis=0)

        # Cost of previous layer activations
        # dc_d(a_L-1); dz_da-1 = self.weights
        return np.dot(dc_dz, self.weights.T)

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.weights -= learning_rate * (self.weights_gradient / batch_size)
        self.biases -= learning_rate * (self.biases_gradient / batch_size)

        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

    def get_output_shape(self):
        return self.units

class Convolution:
    def __init__(self, num_filters, kernel_shape, activation='linear', input_shape=None, stride=1, padding=0):
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.input_size = None
        self.kernel_shape = kernel_shape
        self.kernel_size = np.prod(kernel_shape)
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
        self.dz_da = np.zeros((*self.stacked_output_shape, *self.input_shape))
        for f in range(self.num_filters):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    i_start = i * self.stride
                    j_start = j * self.stride
                    self.dz_da[f, i, j, :, i_start:i_start + self.kernel_shape[0], j_start:j_start + self.kernel_shape[1]] = self.weights[f]
        self.dz_da = np.reshape(self.dz_da, (self.num_filters, self.output_size, -1))

    def init_weights(self, last_layer_shape):
        if self.input_shape is None:
            self.input_shape = last_layer_shape
        if len(self.input_shape) == 2:
            self.input_shape = (1, *self.input_shape)
        self.input_size = np.prod(self.input_shape)
        self.kernel_stack_shape = (self.input_shape[0], *self.kernel_shape)
        self.kernel_stack_size = np.prod(self.kernel_stack_shape)
        self.output_shape = ((self.input_shape[1] - self.kernel_shape[0] + 2 * self.padding) // self.stride + 1, (self.input_shape[2] - self.kernel_shape[1] + 2 * self.padding) // self.stride + 1)
        self.output_size = np.prod(self.output_shape)
        self.stacked_output_shape = (self.num_filters, *self.output_shape)
        self.weights = np.random.randn(self.num_filters, *self.kernel_stack_shape) * np.sqrt(2 / self.kernel_stack_size)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases = np.zeros(self.num_filters)
        self.biases_gradient = np.zeros(self.biases.shape)

        self.precompute_dz_da()

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def get_windows(self, prev_layer_activations, batch_size):
        prev_layer_activations = prev_layer_activations.reshape((batch_size, *self.input_shape))
        prev_layer_activations = np.pad(prev_layer_activations, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        windows = sliding_window_view(prev_layer_activations, window_shape=self.kernel_shape, axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :].transpose((0, 2, 3, 1, 4, 5))
        windows = windows.reshape((batch_size, self.output_size, self.kernel_stack_size))
        return windows

    def forward_pass(self, prev_layer_activations, batch_size):
        windows = self.get_windows(prev_layer_activations, batch_size)
        flattened_weights = self.weights.reshape((self.num_filters, self.kernel_stack_size))
        z_output = np.einsum('bok,fk->bfo', windows, flattened_weights)
        z_output = z_output.reshape((batch_size, self.num_filters, *self.output_shape))
        z_output += self.biases[:, np.newaxis, np.newaxis]
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
        curr_layer_z = curr_layer_z.reshape((batch_size, self.num_filters, self.output_size))
        dc_da = dc_da.reshape((batch_size, self.num_filters, self.output_size))
        da_dz = self.activation.derivative(curr_layer_z)
        if self.activation.elementwise:
            dc_dz = dc_da * da_dz
        else:
            dc_dz = np.einsum('bfo,bfoo->bfo', dc_da, da_dz)

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

        dc_dw = np.einsum('bfo,bok->fk', dc_dz, windows)
        self.weights_gradient += dc_dw.reshape(self.num_filters, *self.kernel_stack_shape)
        self.biases_gradient += np.sum(dc_dz, axis=(0, 2))
        dc_da = np.einsum('bfo,foi->bi', dc_dz, self.dz_da)
        return dc_da

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.weights -= learning_rate * (self.weights_gradient / batch_size)
        self.biases -= learning_rate * (self.biases_gradient / batch_size)

        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

        self.precompute_dz_da()

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
        self.output_size = np.prod(self.output_shape)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        prev_layer_activations = prev_layer_activations.reshape((batch_size, *self.input_shape))
        prev_layer_activations = np.pad(prev_layer_activations, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        # a_output = np.zeros(self.output_shape)
        windows = sliding_window_view(prev_layer_activations, window_shape=self.pool_size, axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        a_output = np.max(windows, axis=(4, 5))
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
        prev_layer_activations = prev_layer_activations.reshape((batch_size, *self.input_shape))
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
        max_areas = (windows == np.max(windows, axis=(4, 5), keepdims=True))
        dc_da = dc_da.reshape((batch_size, *self.output_shape))
        max_areas_scaled = max_areas * dc_da[..., np.newaxis, np.newaxis]
        prev_dc_da = np.zeros_like(prev_layer_activations)
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                i_start = i * self.stride
                j_start = j * self.stride
                prev_dc_da[:, :, i_start:i_start + self.pool_size[0], j_start:j_start + self.pool_size[1]] += max_areas_scaled[:, :, i, j, :, :]
        return prev_dc_da

    def update_weights_and_biases(self, learning_rate, batch_size):
        pass

    def get_output_shape(self):
        return self.output_shape

class Embedding:
    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weights = None
        self.weights_gradient = None
        self.input_shape = None

    def init_weights(self, last_layer_shape):
        self.input_shape = last_layer_shape
        self.weights = np.random.randn(self.vocab_size, self.embedding_size) / np.sqrt(self.embedding_size)
        self.weights_gradient = np.zeros_like(self.weights)

    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        prev_layer_activations = prev_layer_activations.reshape((batch_size, -1))
        a_output = self.weights[prev_layer_activations]

        return a_output, None

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        dc_da = dc_da.reshape((-1, self.embedding_size))

        np.add.at(self.weights_gradient, prev_layer_activations.ravel(), dc_da)

        return None

    def update_weights_and_biases(self, learning_rate, batch_size):
        self.weights -= learning_rate * (self.weights_gradient / batch_size)
        self.weights_gradient = np.zeros_like(self.weights)

    def get_output_shape(self):
        return (self.input_shape, self.embedding_size)

class Attention:
    def __init__(self, d_k, d_v, mask=None):
        self.d_k = d_k
        self.d_v = d_v

        self.sequence_len = None
        self.d_i = None

        self.w_q = None
        self.w_k = None
        self.w_v = None

        self.w_q_gradient = None
        self.w_k_gradient = None
        self.w_v_gradient = None

        self.mask = mask

        self.input_shape = None
        self.output_shape = None



    def init_weights(self, last_layer_shape):
        self.input_shape = last_layer_shape
        self.sequence_len = self.input_shape[0]
        self.d_i = self.input_shape[1]
        self.output_shape = (self.sequence_len, self.d_v)

        self.w_q = np.random.randn(self.d_i, self.d_k) / np.sqrt(2 / self.d_i)
        self.w_k = np.random.randn(self.d_i, self.d_k) / np.sqrt(2 / self.d_i)
        self.w_v = np.random.randn(self.d_i, self.d_v) / np.sqrt(2 / self.d_i)

        self.w_q_gradient = np.zeros_like(self.w_q)
        self.w_k_gradient = np.zeros_like(self.w_k)
        self.w_v_gradient = np.zeros_like(self.w_v)

        # # Upper triangle of -inf so softmax will not let k_i attend to q_j for i > j (dimension is j*i).
        # self.causal_mask = np.triu(np.ones_like((self.sequence_len, self.sequence_len)), k=1) * -1e9


    def get_output(self, prev_layer_activations, batch_size=1):
        a_output, z_output = self.forward_pass(prev_layer_activations, batch_size)
        return a_output

    def forward_pass(self, prev_layer_activations, batch_size):
        # t and s are both sequence_length, this is just to distinguish the q dimension and the k dimension for einsum.
        q = np.einsum('ik,bti->btk', self.w_q, prev_layer_activations)
        k = np.einsum('ik,bsi->bsk', self.w_k, prev_layer_activations)
        v = np.einsum('iv,bsi->bsv', self.w_v, prev_layer_activations)

        raw_scores = np.einsum('btk,bsk->bts', q, k) / np.sqrt(self.d_k)
        raw_scores = np.where(self.mask, -1e9, raw_scores)

        attention_scores = activations.softmax(raw_scores)

        a_output = np.einsum('bts,bsv->btv', attention_scores, v)

        return a_output, (raw_scores, attention_scores, q, k, v)

    def backward_pass(self, prev_layer_activations, curr_layer_z, dc_da, batch_size):
        raw_scores, attention_scores, q, k, v = curr_layer_z
        dc_dv = np.einsum('bsv,bst->btv', dc_da, attention_scores)
        self.w_v_gradient += np.einsum('bsv,bsi->iv', dc_dv, prev_layer_activations)

        dc_dattention_scores = np.einsum('bsv,btv->bts', dc_da, v)
        dattention_scores_draw_scores = activations.transformer_softmax_derivative(raw_scores)

        dc_draw_scores = np.einsum('bts,btss->bts', dc_dattention_scores, dattention_scores_draw_scores)
        if self.mask is not None:
            dc_draw_scores = np.where(self.mask, 0, dc_draw_scores)

        dc_dq = np.einsum('bts,bsk->btk', dc_draw_scores, k) / np.sqrt(self.d_k)
        dc_dk = np.einsum('bts,btk->bsk', dc_draw_scores, q) / np.sqrt(self.d_k)

        self.w_q_gradient += np.einsum('btk,bti->ik', dc_dq, prev_layer_activations)
        self.w_k_gradient += np.einsum('bsk,bsi->ik', dc_dk, prev_layer_activations)

        dc_dprev_layer_activations = np.einsum('btk,ik->bti', dc_dq, self.w_q) + np.einsum('bsk,ik->bsi', dc_dk, self.w_k) + np.einsum('bsv,iv->bsi', dc_dv, self.w_v)
        return dc_dprev_layer_activations


    def update_weights_and_biases(self, learning_rate, batch_size):
        self.w_q -= learning_rate * (self.w_q_gradient / batch_size)
        self.w_k -= learning_rate * (self.w_k_gradient / batch_size)
        self.w_v -= learning_rate * (self.w_v_gradient / batch_size)

        self.w_q_gradient = np.zeros_like(self.w_q)
        self.w_k_gradient = np.zeros_like(self.w_k)
        self.w_v_gradient = np.zeros_like(self.w_v)

    def get_output_shape(self):
        return self.output_shape
