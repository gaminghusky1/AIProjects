import numpy as np
from base_model import loss_functions
from base_model import optimizers
from pathlib import Path
import pickle

def to_ohe(y_batch, vocab_size):
    b, t = y_batch.shape
    ohe = np.zeros((b, t, vocab_size), dtype=np.float32)
    ohe[np.arange(b)[:, np.newaxis], np.arange(t), y_batch] = 1.0
    return ohe

class Model:
    def __init__(self, input_shape, *layers):
        self.input_shape = input_shape
        self.layers = layers
        self.loss_func_name = None
        self.loss_func = None
        self.optimizer = None
        self.compiled = False
        self.curr_loss = 0
        self.curr_accuracy = 0

    def compile(self, loss='mse', optimizer=None):
        self.loss_func_name = loss
        self.loss_func = loss_functions.LossFunction(loss)
        self.optimizer = optimizers.optimizer_dict[optimizer]() if optimizer is not None else None
        self.compiled = True

    def forward_propagate(self, inputs, batch_size):
        z_outputs = []
        a_outputs = [inputs]
        for layer in self.layers:
            a_output, z_output = layer.forward_pass(a_outputs[-1], batch_size)
            a_outputs.append(a_output)
            z_outputs.append(z_output)

        return a_outputs, z_outputs

    def backward_propagate(self, a_outputs, z_outputs, y, batch_size):
        dc_da = self.loss_func.derivative(y, a_outputs[-1])
        for i in reversed(range(len(self.layers))):
            dc_da = self.layers[i].backward_pass(a_outputs[i], z_outputs[i], dc_da, batch_size)

    def fit(self, x, y, epochs, learning_rate=0.01, batch_size=1, shuffle=True, verbose=1, y_ohe=True, save_after_each_epoch=False, path="model"):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before fitting.")

        self.optimizer.learning_rate = learning_rate
        if verbose >= 0:
            print(f"Training model with {epochs} epochs and learning rate of {learning_rate}...")

        last_layer_shape = self.input_shape
        for layer in self.layers:
            layer.init_weights(last_layer_shape)
            last_layer_shape = layer.get_output_shape()

        data_len = len(x)
        for i in range(epochs):
            if shuffle:
                indices = np.arange(data_len)
                np.random.shuffle(indices)
                x, y = x[indices], y[indices]

            loss_sum = 0
            num_correct = 0

            idx = 0
            while idx < data_len:
                curr_batch_size = min(batch_size, data_len - idx)
                x_batch = x[idx:idx + curr_batch_size]
                y_batch = y[idx:idx + curr_batch_size]

                if not y_ohe:
                    y_batch = to_ohe(y_batch, last_layer_shape[-1])

                a_outputs, z_outputs = self.forward_propagate(x_batch, curr_batch_size)
                loss_sum += np.sum(self.loss_func(y_batch, a_outputs[-1])) * curr_batch_size
                num_correct += np.mean(np.argmax(a_outputs[-1], axis=-1) == np.argmax(y_batch, axis=-1)) * curr_batch_size

                self.backward_propagate(a_outputs, z_outputs, y_batch, curr_batch_size)

                if self.optimizer is not None:
                    params = []
                    grads = []
                    for layer in self.layers:
                        params.extend(layer.get_params())
                        grads.extend(layer.get_grads())
                    self.optimizer.step(params, grads)
                else:
                    for layer in self.layers:
                        layer.update_weights_and_biases(learning_rate, curr_batch_size)

                for layer in self.layers:
                    layer.reset_grads()

                idx += curr_batch_size
                if verbose > 1:
                    print(f"Epoch: {i + 1}/{epochs}; Batch: {idx // batch_size + (idx % batch_size != 0)}/{(data_len + batch_size - 1) // batch_size}; Loss: {loss_sum / idx:.5f}; Accuracy: {num_correct / idx:.5f}")

            self.curr_loss = loss_sum / data_len
            self.curr_accuracy = num_correct / data_len
            if verbose > 0:
                print(f"Epoch {i+1}/{epochs} finished with loss of {loss_sum / data_len:.5f} and accuracy of {num_correct / data_len:.5f}")
            if save_after_each_epoch:
                self.save_as(path + f"_epoch_{i+1}")
        if verbose >= 0:
            print("Training completed.")

    def predict(self, x):
        y_hat = x
        for layer in self.layers:
            y_hat = layer.get_output(y_hat)
        return y_hat

    def test(self, x, y, y_ohe=True):
        data_len = len(x)
        y_hat = x
        for layer in self.layers:
            y_hat = layer.get_output(y_hat, data_len)
        if not y_ohe:
            y = to_ohe(y, y_hat.shape[-1])
        num_correct = np.mean(np.argmax(y_hat, axis=-1) == np.argmax(y, axis=-1)) * data_len
        return num_correct / data_len

    def get_param_count(self):
        param_count = 0
        for layer in self.layers:
            param_count += layer.get_param_count()
        return param_count

    def get_current_loss(self):
        return self.curr_loss

    def get_current_accuracy(self):
        return self.curr_accuracy

    def save_as(self, path):
        print(f"Saving model to {path}...")
        full_path = Path(path).with_suffix(".pkl")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved to {path}.")

    @classmethod
    def load_from(cls, path):
        with open(path + ".pkl", "rb") as file:
            return pickle.load(file)