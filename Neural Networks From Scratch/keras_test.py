import keras
from keras.datasets import mnist, cifar10
import numpy as np

def load_mnist():
    # Read in dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return np.eye(classes)[y]

def main():
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_mnist()
    x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_cifar()

    # y_train_mnist_oh = one_hot(y_train_mnist)
    # y_test_mnist_oh = one_hot(y_test_mnist)

    # cifar model
    keras_model = keras.Sequential([
        keras.layers.Input((32, 32, 3)),

        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomTranslation(0.1, 0.1),

        keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")
    ])

    # mnist model
    # keras_model = keras.Sequential([
    #     keras.layers.Input((28, 28, 1)),
    #     # keras.layers.Conv2D(32, (3, 3), activation="relu"),
    #     # keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    #     # keras.layers.Conv2D(64, (3, 3), activation="relu"),
    #     keras.layers.Flatten(),
    #     # keras.layers.Dense(1024, activation="relu"),
    #     keras.layers.Dense(512, activation="relu"),
    #     keras.layers.Dense(256, activation="relu"),
    #     keras.layers.Dense(128, activation="relu"),
    #     # keras.layers.Dense(64, activation="relu"),
    #     keras.layers.Dense(10, activation="softmax")
    # ])

    keras_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'], jit_compile=True)

    # keras_model.fit(x_train_mnist, y_train_mnist, epochs=15, batch_size=32, verbose=1)
    # keras_model.evaluate(x_test_mnist, y_test_mnist, verbose=1)

    keras_model.fit(x_train_cifar, y_train_cifar, epochs=15, batch_size=256, verbose=1, validation_data=(x_test_cifar, y_test_cifar))
    keras_model.evaluate(x_test_cifar, y_test_cifar)

if __name__ == '__main__':
    main()