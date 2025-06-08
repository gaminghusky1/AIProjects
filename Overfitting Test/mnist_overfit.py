import numpy as np
from keras.datasets import mnist
import keras
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf


def load_random_images(filepath="random_images.pkl"):
    with open(filepath, "rb") as f:
        images, labels = pickle.load(f)
    return images, labels

def format_elapsed_time(elapsed_time):
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, secs = divmod(remainder, 60)
    secs, ms = divmod(secs, 1)
    ms *= 1000

    parts = []
    display = False
    if hours:
        parts.append(f"{hours:.0f}h")
        display = True
    if minutes or display:
        parts.append(f"{minutes:.0f}m")
        display = True
    if secs or display:
        parts.append(f"{secs:.0f}s")
        display = True
    if ms or display or not parts:
        parts.append(f"{ms:.2f}ms")

    return " ".join(parts)

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_small, y_train_small = load_random_images()

    y_train = np.eye(10)[y_train]
    y_train_small = np.eye(10)[y_train_small]
    y_test = np.eye(10)[y_test]
    step_size = 16
    start = 16
    end = 128
    df = pd.DataFrame(columns=["layer_size", "param_num", "accuracy_difference"])
    for layer_size in range(start, end + 1, step_size):
        start_time = time.perf_counter()
        print(f"Training model number {(layer_size - start) // step_size + 1} of {(end - start) // step_size + 1} with layer size {layer_size}...")
        model = keras.Sequential([
            keras.layers.Input((28, 28, 1)),

            keras.layers.Conv2D(layer_size, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(layer_size, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train_small, y_train_small, epochs=20, batch_size=16, verbose=0)

        train_accuracy = model.evaluate(x_train_small, y_train_small)[1]
        test_accuracy = model.evaluate(x_test, y_test)[1]

        elapsed_time = time.perf_counter() - start_time
        print(f"Finished training model number {(layer_size - start) // step_size + 1} of {(end - start) // step_size + 1} in {format_elapsed_time(elapsed_time)} ({(layer_size - start + step_size) / (end - start + step_size) * 100:.2f}% done)")
        print('Train accuracy:', f"{train_accuracy * 100:.2f}%")
        print('Test accuracy:', f"{test_accuracy * 100:.2f}%")
        print('Difference (train - test):', f"{(train_accuracy - test_accuracy) * 100:.2f}%")
        print()
        idx = len(df)
        df.loc[idx, "layer_size"] = layer_size
        df.loc[idx, "param_num"] = model.count_params()
        df.loc[idx, "accuracy_difference"] = train_accuracy - test_accuracy

    print(df)
    # df.plot.scatter(x="layer_size", y="accuracy_difference", title="Layer Size vs. Difference in Accuracy", alpha=0.5)
    df.to_csv(f"accuracy_diff_{start}_to_{end}_with_step_size_{step_size}.csv")

if __name__ == '__main__':
    main()