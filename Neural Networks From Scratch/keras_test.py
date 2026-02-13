import keras
from keras.datasets import mnist
import numpy as np

def load_mnist():
    # Read in dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return np.eye(classes)[y]

def main():
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_mnist()

    y_train_mnist_oh = one_hot(y_train_mnist)
    y_test_mnist_oh = one_hot(y_test_mnist)
    keras_model = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        # keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    keras_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    keras_model.fit(x_train_mnist, y_train_mnist_oh, epochs=30, batch_size=32, verbose=1)
    keras_model.evaluate(x_test_mnist, y_test_mnist_oh, verbose=1)

if __name__ == '__main__':
    main()