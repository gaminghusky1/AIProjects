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
        last_layer_shape = self.input_shape
        for layer in self.layers:
            layer.init_weights(last_layer_shape)
            last_layer_shape = layer.get_output_shape()
        self.compiled = True

    def forward_propagate(self, inputs):
        z_outputs = []
        a_outputs = [inputs]
        for layer in self.layers:
            a_output, z_output = layer.forward_pass(a_outputs[-1])
            a_outputs.append(a_output)
            z_outputs.append(z_output)

        return a_outputs, z_outputs

    def backward_propagate(self, a_outputs, z_outputs, y):
        dc_da = self.loss_func.derivative(y, a_outputs[-1])
        for i in reversed(range(len(self.layers))):
            dc_da = self.layers[i].backward_pass(a_outputs[i], z_outputs[i], dc_da)

    def fit(self, x, y, epochs, learning_rate=0.01, batch_size=1, shuffle=True, output_progress=True):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before fitting.")

        if output_progress:
            print(f"Training model with {epochs} epochs and learning rate of {learning_rate:.2f}...")

        data_len = len(x)
        for i in range(epochs):
            if shuffle:
                indices = np.arange(data_len)
                np.random.shuffle(indices)
                x, y = x[indices], y[indices]

            loss_sum = 0
            num_correct = 0
            for j in range(data_len):
                a_outputs, z_outputs = self.forward_propagate(x[j])
                loss_sum += self.loss_func(y[j], a_outputs[-1])
                num_correct += np.argmax(a_outputs[-1]) == np.argmax(y[j])

                self.backward_propagate(a_outputs, z_outputs, y[j])
                if j != 0 and (j + 1) % batch_size == 0:
                    for layer in self.layers:
                        layer.update_weights_and_biases(learning_rate, batch_size)

            if data_len % batch_size != 0:
                for layer in self.layers:
                    layer.update_weights_and_biases(learning_rate, data_len % batch_size)

            if output_progress:
                print(f"Epoch {i+1}/{epochs} finished with loss of {loss_sum / data_len:.5f} and accuracy of {num_correct / data_len:.5f}")
        if output_progress:
            print("Training completed.")

    def predict(self, x):
        y_hat = x
        for layer in self.layers:
            y_hat = layer.get_output(y_hat)
        return y_hat

    def test(self, x, y):
        num_correct = 0
        for i in range(len(x)):
            num_correct += np.argmax(self.predict(x[i])) == np.argmax(y[i])
        return num_correct / len(x)

    def save_as(self, path):
        with open(path + ".pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_from(cls, path):
        with open(path + ".pkl", "rb") as file:
            return pickle.load(file)