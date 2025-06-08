import numpy as np
from keras.datasets import mnist
import layers
import model
import pickle

def load_random_images(filepath="random_images.pkl"):
    with open(filepath, "rb") as f:
        images, labels = pickle.load(f)
    return images, labels

def load():
    # Read in dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape into vectors
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return np.eye(classes)[y].T

def main():
    x_train, y_train, x_test, y_test = load()
    x_train_small, y_train_small = load_random_images()
    x_test_small, y_test_small = load_random_images("random_images_test.pkl")

    y_train_oh = one_hot(y_train).T
    y_test_oh = one_hot(y_test).T
    y_train_small_oh = one_hot(y_train_small).T
    y_test_small_oh = one_hot(y_test_small).T

    test_model = model.Model(
        (1, 28, 28),
        layers.Convolution(16, (3, 3), activation="relu"),
        layers.MaxPooling((2, 2)),
        layers.Dense(16, activation="relu"),
        layers.Dense(10, activation="softmax")
    )

    test_model.compile(loss="categorical_crossentropy")

    test_model.fit(x_train_small, y_train_small_oh, epochs=10, learning_rate=0.01)
    print(test_model.test(x_test_small, y_test_small_oh))

    test_model.save_as("convolutional_model")


if __name__ == "__main__":
    main()