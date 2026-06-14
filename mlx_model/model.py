import time

import mlx.core as mx
import pandas as pd
from mlx_model import loss_functions
from mlx_model import optimizers
from pathlib import Path
import pickle

def to_ohe(y_batch, vocab_size):
    b, t = y_batch.shape
    ohe = mx.zeros((b, t, vocab_size), dtype=mx.float32)
    ohe[mx.arange(b)[:, mx.newaxis], mx.arange(t), y_batch] = 1.0
    return ohe

def default_lr_function(step):
    return 1

class Model:
    def __init__(self, input_shape, *layers):
        self.input_shape = input_shape
        self.layers = layers
        self.loss_func_name = None
        self.loss_func = None
        self.optimizer = None
        self.metrics = None
        self.compiled = False
        self.curr_loss = 0
        self.curr_accuracy = 0

    def compile(self, loss='mse', optimizer=None, optimizer_kwargs=None):
        self.loss_func_name = loss
        self.loss_func = loss_functions.LossFunction(loss)
        self.optimizer = optimizers.optimizer_dict[optimizer](**(optimizer_kwargs or {}))
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

    def fit(self, x, y, epochs, learning_rate=0.01, lr_function=default_lr_function, augmentation_function=None, curr_step=0, batch_size=1, ema_beta=0.99, shuffle=True, verbose=1, ohe_y=False, previously_trained_epochs=0, save_after_num_epochs=-1, model_save_path="model", save_metrics=False):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before fitting.")

        if verbose >= 0:
            print(f"Training model with {epochs} epochs and learning rate of {learning_rate}...")

        fit_start_time = time.perf_counter()

        ema_loss = None
        if save_metrics:
            try:
                self.metrics = pd.read_csv(model_save_path + "_metrics.csv", index_col=0)
                curr_step = len(self.metrics)
                ema_loss = self.metrics.loc[curr_step - 1]["ema_loss"]
            except FileNotFoundError:
                self.metrics = pd.DataFrame(columns=["loss", "ema_loss", "accuracy"])

        data_len = len(x)
        for i in range(previously_trained_epochs, epochs):
            epoch_start_time = time.perf_counter()
            if shuffle:
                indices = mx.arange(data_len)
                indices = mx.random.permutation(indices)
                x, y = x[indices], y[indices]

            epoch_loss_sum = 0
            epoch_num_correct = 0

            idx = 0
            while idx < data_len:
                batch_start_time = time.perf_counter()
                curr_batch_size = min(batch_size, data_len - idx)
                # x_batch = mx.array(x[idx:idx + curr_batch_size])
                # y_batch = mx.array(y[idx:idx + curr_batch_size])
                x_batch = x[idx:idx + curr_batch_size]
                y_batch = y[idx:idx + curr_batch_size]

                if augmentation_function is not None:
                    x_batch = mx.stack([augmentation_function(x_batch[i]) for i in range(x_batch.shape[0])], axis=0)

                if ohe_y:
                    y_batch = to_ohe(y_batch, self.layers[-1].get_output_shape()[-1])

                a_outputs, z_outputs = self.forward_propagate(x_batch)
                batch_loss_sum = self.loss_func(y_batch, a_outputs[-1]) * curr_batch_size
                if self.loss_func_name in {"sparse_softmax_crossentropy"}:
                    batch_num_correct = mx.mean(mx.argmax(a_outputs[-1], axis=-1) == y_batch) * curr_batch_size
                else:
                    batch_num_correct = mx.mean(mx.argmax(a_outputs[-1], axis=-1) == mx.argmax(y_batch, axis=-1)) * curr_batch_size


                self.backward_propagate(a_outputs, z_outputs, y_batch)

                self.optimizer.learning_rate = learning_rate * lr_function(curr_step)
                param_refs = []
                grads = []
                for layer in self.layers:
                    param_refs.extend(layer.get_param_refs())
                    grads.extend(layer.get_grads())
                self.optimizer.step(param_refs, grads)

                for layer in self.layers:
                    layer.reset_grads()

                idx += curr_batch_size
                mx.eval(batch_loss_sum, batch_num_correct)

                if ema_loss is None:
                    ema_loss = batch_loss_sum.item() / curr_batch_size
                else:
                    ema_loss = ema_beta * ema_loss + (1 - ema_beta) * batch_loss_sum.item() / curr_batch_size
                if save_metrics:
                    self.metrics.loc[curr_step] = {"loss": batch_loss_sum.item() / curr_batch_size, "ema_loss": ema_loss, "accuracy": batch_num_correct.item() / curr_batch_size}
                epoch_loss_sum += batch_loss_sum.item()
                epoch_num_correct += batch_num_correct.item()
                curr_step += 1
                if verbose > 1:
                    batch_time = time.perf_counter() - batch_start_time
                    print(f"Epoch {i + 1}/{epochs}, Batch {idx // batch_size + (idx % batch_size != 0)}/{(data_len + batch_size - 1) // batch_size}; Loss: {batch_loss_sum.item() / curr_batch_size:.5f}; Acc: {batch_num_correct.item() / curr_batch_size:.5f}; EMA Loss: {ema_loss:.5f}; Avg Loss: {epoch_loss_sum / idx:.5f}; Avg Acc: {epoch_num_correct / idx:.5f}; Time: {batch_time:.5f}s")

            self.curr_loss = epoch_loss_sum / data_len
            self.curr_accuracy = epoch_num_correct / data_len
            if verbose > 0:
                epoch_time = time.perf_counter() - epoch_start_time
                print(f"Epoch {i+1}/{epochs} finished in {epoch_time:.5f} seconds with average loss of {epoch_loss_sum / data_len:.5f} and average accuracy of {epoch_num_correct / data_len:.5f}")
            if save_after_num_epochs > 0 and (i + 1) % save_after_num_epochs == 0:
                self.save_as(model_save_path + f"_epoch_{i+1}")
                if save_metrics:
                    self.metrics.to_csv(model_save_path + "_metrics.csv")
        if verbose >= 0:
            fit_time = time.perf_counter() - fit_start_time
            print(f"Training completed in {fit_time:.5f} seconds.")
        if save_metrics:
            self.metrics.to_csv(model_save_path + "_metrics.csv")

    def predict(self, x):
        y_hat = x
        for layer in self.layers:
            y_hat = layer.get_output(y_hat)
        return y_hat

    def test(self, x, y, ohe_y=False):
        y_hat = x
        for layer in self.layers:
            y_hat = layer.get_output(y_hat)
        if ohe_y:
            y = to_ohe(y, y_hat.shape[-1])
        if self.loss_func_name in {"sparse_softmax_crossentropy"}:
            return mx.mean(mx.argmax(y_hat, axis=-1) == y).item()
        else:
            return mx.mean(mx.argmax(y_hat, axis=-1) == mx.argmax(y, axis=-1)).item()

    def test_loss(self, x, y, ohe_y=False):
        y_hat = x
        for layer in self.layers:
            y_hat = layer.get_output(y_hat)
        if ohe_y:
            y = to_ohe(y, y_hat.shape[-1])
        return self.loss_func(y, y_hat).item()

    def get_param_count(self):
        if not self.compiled:
            raise RuntimeError("Model must be compiled to know parameter count.")
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
