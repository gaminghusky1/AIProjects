import numpy as np
import loss_functions
import pickle

class Model:
    def __init__(self, input_shape, *layers):
        self.input_shape = input_shape
        self.layers = layers
        self.loss_func_name = None
        self.loss_func = None
        self.compiled = False

    def compile(self, loss='mse'):
        self.loss_func_name = loss
        self.loss_func = loss_functions.LossFunction(loss)
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

    def fit(self, x, y, epochs, learning_rate=0.01, batch_size=1, shuffle=True, verbose=1):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before fitting.")

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

                a_outputs, z_outputs = self.forward_propagate(x_batch, curr_batch_size)
                loss_sum += np.sum(self.loss_func(y_batch, a_outputs[-1])) * curr_batch_size
                num_correct += np.mean(np.argmax(a_outputs[-1], axis=-1) == np.argmax(y_batch, axis=-1)) * curr_batch_size

                self.backward_propagate(a_outputs, z_outputs, y_batch, curr_batch_size)

                for layer in self.layers:
                    layer.update_weights_and_biases(learning_rate, curr_batch_size)

                idx += curr_batch_size
                if verbose > 1:
                    print(f"Epoch: {i + 1}; Batch: {idx // batch_size + (idx % batch_size != 0)}; Loss: {loss_sum / idx:.5f}; Accuracy: {num_correct / idx:.5f}")

            if verbose > 0:
                print(f"Epoch {i+1}/{epochs} finished with loss of {loss_sum / data_len:.5f} and accuracy of {num_correct / data_len:.5f}")
        print("Training completed.")

    def predict(self, x):
        y_hat = x
        for layer in self.layers:
            y_hat = layer.get_output(y_hat)
        return y_hat

    def test(self, x, y):
        data_len = len(x)
        y_hat = x
        for layer in self.layers:
            y_hat = layer.get_output(y_hat, data_len)
        num_correct = np.sum(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))
        return num_correct / data_len

    def save_as(self, path):
        print(f"Saving model to {path}...")
        with open(path + ".pkl", "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved to {path}.")

    @classmethod
    def load_from(cls, path):
        with open(path + ".pkl", "rb") as file:
            return pickle.load(file)