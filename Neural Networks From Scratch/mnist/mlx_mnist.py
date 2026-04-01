import mlx.core as mx
from keras.datasets import mnist
from mlx_model import *

def load_mnist():
    # Read in dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = mx.array(x_train)
    y_train = mx.array(y_train)
    x_test = mx.array(x_test)
    y_test = mx.array(y_test)

    # Reshape into vectors
    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    mx.eval(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return mx.eye(classes)[y]

def main():
    x_train, y_train, x_test, y_test = load_mnist()

    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    # mnist_model = model.Model(
    #     (1, 28, 28),
    #     # layers.Reshape((1, 28, 28)),
    #     # layers.Convolution(32, (3, 3), activation="relu"),
    #     # layers.MaxPooling((2, 2), stride=2),
    #     # layers.Convolution(64, (3, 3), activation="relu"),
    #     # layers.MaxPooling((2, 2)),
    #     layers.Flatten(),
    #     # layers.Dense(1024, activation="relu"),
    #     # layers.Dense(512, activation="relu"),
    #     # layers.Dense(256, activation="relu"),
    #     layers.Dense(128, activation="relu"),
    #     # layers.Dense(64, activation="relu"),
    #     layers.Dense(10, activation="crossentropy_softmax")
    # )

    # mnist_model.compile(loss="softmax_crossentropy", optimizer="adam")

    mnist_model = model.Model.load_from("Models/test_epoch_1")

    print("Param Count:", mnist_model.get_param_count())

    mnist_model.fit(x_train, y_train_oh, epochs=1, learning_rate=0.001, batch_size=32, verbose=2, save_after_num_epochs=1, model_save_path="Models/test", save_metrics=True)
    print("Accuracy on test dataset: " + str(mnist_model.test(x_test, y_test_oh)))

    # mnist_model.save_as("Models/mnist_model")

if __name__ == "__main__":
    main()