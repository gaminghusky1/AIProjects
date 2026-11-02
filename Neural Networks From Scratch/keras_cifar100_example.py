import keras
from keras.datasets import cifar100
import numpy as np

def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return np.eye(classes)[y]

def main():
    x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_cifar()

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
        keras.layers.Dense(100, activation="softmax")
    ])

    keras_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'], jit_compile=True)

    keras_model.fit(x_train_cifar, y_train_cifar, epochs=15, batch_size=128, verbose=1, validation_data=(x_test_cifar, y_test_cifar))
    keras_model.evaluate(x_test_cifar, y_test_cifar)

    # Save the model
    keras_model.save("cifar_model.keras")

    # To load the model again, run:
    # keras_model = keras.models.load_model("cifar_model.keras")

if __name__ == '__main__':
    main()