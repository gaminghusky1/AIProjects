import mlx.core as mx
from keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from mlx_model import *

def load_mnist():
    # Read in dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = mx.array(x_train)
    y_train = mx.array(y_train)
    x_test = mx.array(x_test)
    y_test = mx.array(y_test)

    # Reshape into vectors
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    mx.eval(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test

def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = mx.array(x_train)
    y_train = mx.array(y_train)
    x_test = mx.array(x_test)
    y_test = mx.array(y_test)

    x_train = mx.transpose(x_train, (0, 3, 1, 2))
    x_test = mx.transpose(x_test, (0, 3, 1, 2))

    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return mx.eye(classes)[y]

def main():
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_mnist()
    x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_cifar()

    y_train_mnist_oh = one_hot(y_train_mnist)
    y_test_mnist_oh = one_hot(y_test_mnist)
    y_train_cifar_oh = one_hot(y_train_cifar)
    y_test_cifar_oh = one_hot(y_test_cifar)

    # test_model = model.Model(
    #     (3, 32, 32),
    #     layers.Convolution(32, (3, 3), activation="relu"),
    #     layers.MaxPooling((2, 2)),
    #     layers.Convolution(32, (3, 3), activation="relu"),
    #     layers.MaxPooling((2, 2)),
    #     layers.Dense(512, activation="relu"),
    #     layers.Dense(256, activation="relu"),
    #     layers.Dense(128, activation="relu"),
    #     layers.Dense(64, activation="relu"),
    #     layers.Dense(10, activation="softmax")
    # )

    test_model = model.Model(
        (1, 28, 28),
        # layers.Dense(1024, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="crossentropy_softmax")
    )

    test_model.compile(loss="softmax_crossentropy", optimizer="adam")

    # test_model.fit(x_train_cifar, y_train_cifar_oh, epochs=15, learning_rate=0.01, batch_size=32, verbose=2)
    # print("Accuracy on test dataset: " + str(test_model.test(x_test_cifar, y_test_cifar_oh)))

    test_model.fit(x_train_mnist, y_train_mnist_oh, epochs=15, learning_rate=0.01, batch_size=32, verbose=2)
    print("Accuracy on test dataset: " + str(test_model.test(x_test_mnist, y_test_mnist_oh)))

    # test_model.save_as("Models/mnist_batched_model")

if __name__ == "__main__":
    main()