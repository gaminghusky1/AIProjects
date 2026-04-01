import mlx.core as mx
from tensorflow.keras.datasets import cifar10
from mlx_model import *
import math

def cosine_lr(step, total_steps=1563*50, lr_max=0.1/0.001, lr_min=1e-4/0.001, warmup_steps=int(1563*50*0.05)):
    """
    Cosine learning rate schedule with optional linear warmup.

    Args:
        step (int): current step (starts at 0)
        total_steps (int): total number of training steps
        lr_max (float): peak learning rate
        lr_min (float): minimum learning rate at end
        warmup_steps (int): number of warmup steps

    Returns:
        float: learning rate for this step
    """

    # Warmup phase
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps

    # Cosine decay phase
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)  # clamp for safety

    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    return lr_min + (lr_max - lr_min) * cosine_decay

def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = mx.array(x_train)
    y_train = mx.array(y_train)
    x_test = mx.array(x_test)
    y_test = mx.array(y_test)

    x_train = mx.transpose(x_train, (0, 3, 2, 1))
    x_test = mx.transpose(x_test, (0, 3, 2, 1))

    # x_train = x_train.reshape(50000, 3072)
    # x_test = x_test.reshape(10000, 3072)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return mx.eye(classes)[y]

def main():
    x_train, y_train, x_test, y_test = load_cifar()

    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    cifar_model = model.Model(
        (3, 32, 32),
        layers.Convolution(32, (3, 3), activation="relu", padding=1),
        layers.Convolution(32, (3, 3), activation="relu", padding=1),
        layers.MaxPooling((2, 2), stride=2),
        layers.Dropout(0.2),
        layers.Convolution(64, (3, 3), activation="relu", padding=1),
        layers.Convolution(64, (3, 3), activation="relu", padding=1),
        layers.MaxPooling((2, 2), stride=2),
        layers.Dropout(0.3),
        layers.Convolution(128, (3, 3), activation="relu", padding=1),
        layers.Convolution(128, (3, 3), activation="relu", padding=1),
        layers.MaxPooling((2, 2), stride=2),

        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),

        layers.Dropout(0.5),
        layers.Dense(10, activation="crossentropy_softmax")
    )

    # cifar_model = model.Model.load_from("Models/cifar_conv")

    cifar_model.compile(loss="softmax_crossentropy", optimizer="adam")

    print("Param Count:", cifar_model.get_param_count())

    cifar_model.fit(x_train, y_train_oh, epochs=50, learning_rate=0.001, lr_function=cosine_lr, batch_size=32, verbose=2, save_after_num_epochs=10, model_save_path="Models/cifar_conv", save_metrics=True)
    # print("Accuracy on test dataset: " + str(cifar_model.test(x_test, y_test_oh)))

    cifar_model.save_as("Models/cifar_conv_final")

if __name__ == "__main__":
    main()