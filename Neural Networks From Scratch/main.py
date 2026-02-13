import numpy as np
from keras.datasets import mnist
import keras
from tensorflow.keras.datasets import cifar10
import layers
import model
import pickle

def load_random_images(filepath="random_images.pkl"):
    with open(filepath, "rb") as f:
        images, labels = pickle.load(f)
    return images, labels

def load_mnist():
    # Read in dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape into vectors
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return np.eye(classes)[y]

def main():
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_mnist()
    x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_cifar()
    x_train_small, y_train_small = load_random_images()
    x_test_small, y_test_small = load_random_images("random_images_test.pkl")


    y_train_mnist_oh = one_hot(y_train_mnist)
    y_test_mnist_oh = one_hot(y_test_mnist)
    y_train_cifar_oh = one_hot(y_train_cifar)
    y_test_cifar_oh = one_hot(y_test_cifar)
    y_train_small_oh = one_hot(y_train_small)
    y_test_small_oh = one_hot(y_test_small)

    # test_model = model.Model(
    #     (3, 32, 32),
    #     # layers.Convolution(32, (3, 3), activation="relu"),
    #     # layers.MaxPooling((2, 2)),
    #     # layers.Convolution(32, (3, 3), activation="relu"),
    #     # layers.MaxPooling((2, 2)),
    #     # layers.Dense(512, activation="relu"),
    #     # layers.Dense(256, activation="relu"),
    #     layers.Dense(128, activation="relu"),
    #     layers.Dense(64, activation="relu"),
    #     layers.Dense(10, activation="softmax")
    # )

    test_model = model.Model(
        (1, 28, 28),
        # layers.Convolution(32, (3, 3), activation="relu"),
        # layers.MaxPooling((2, 2)),
        # layers.Convolution(32, (3, 3), activation="relu"),
        # layers.MaxPooling((2, 2)),
        # layers.Dense(512, activation="relu"),
        # layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    )

    test_model.compile(loss="categorical_crossentropy")
    # test_model.compile(loss="mse")

    # test_model.fit(x_train_cifar, y_train_cifar_oh, epochs=15, learning_rate=0.01, batch_size=16, verbose=1)
    # print("Accuracy on test dataset: " + str(test_model.test(x_test_cifar, y_test_cifar_oh)))

    test_model.fit(x_train_mnist, y_train_mnist_oh, epochs=15, learning_rate=0.01, batch_size=1, verbose=1)
    print("Accuracy on test dataset: " + str(test_model.test(x_test_mnist, y_test_mnist_oh)))

    # test_model.save_as("Models/mnist_batched_model")

if __name__ == "__main__":
    main()