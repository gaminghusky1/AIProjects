import numpy as np
from tensorflow.keras.datasets import cifar10
from base_model import *
import math

def warmup_cosine_lr(curr_step, total_steps=782*200, warmup_steps=int(0.01*782*200), min_lr_ratio=0.05):
    if curr_step < warmup_steps:
        return curr_step / warmup_steps  # linear warmup

    progress = (curr_step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine

def random_horizontal_flip_chw_mx(image, p=0.5):
    """
    image shape: (C, H, W)
    Flips left-right with probability p.
    """
    do_flip = np.random.uniform() < p
    return image[:, :, ::-1] if bool(do_flip) else image


def random_crop_with_padding_chw_mx(image, crop_size=32, padding=4, pad_value=0.0):
    """
    image shape: (C, H, W)
    Pads H/W, then randomly crops back to crop_size x crop_size.
    """
    c, h, w = image.shape

    padded_h = h + 2 * padding
    padded_w = w + 2 * padding

    padded = np.full((c, padded_h, padded_w), pad_value, dtype=image.dtype)
    padded[:, padding:padding + h, padding:padding + w] = image

    max_y = padded_h - crop_size
    max_x = padded_w - crop_size

    y = int(np.random.randint(0, max_y + 1))
    x = int(np.random.randint(0, max_x + 1))

    return padded[:, y:y + crop_size, x:x + crop_size]


def cifar_augmentation_chw_mx(image):
    """
    Standard CIFAR-style augmentation for a single (C, H, W) image.
    """
    image = random_horizontal_flip_chw_mx(image, p=0.5)
    image = random_crop_with_padding_chw_mx(image, crop_size=32, padding=4)
    return image

def load_cifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = np.transpose(x_train, (0, 3, 2, 1))
    x_test = np.transpose(x_test, (0, 3, 2, 1))

    # x_train = x_train.reshape(50000, 3072)
    # x_test = x_test.reshape(10000, 3072)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y, classes=10):
    return np.eye(classes)[y]

def main():
    x_train, y_train, x_test, y_test = load_cifar()

    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    cifar_model = model.Model(
        (3, 32, 32),
        layers.Convolution(64, (3, 3), activation="relu", padding=1),
        layers.Convolution(64, (3, 3), activation="relu", padding=1),
        layers.MaxPooling((2, 2), stride=2),
        layers.Dropout(0.1),
        layers.Convolution(128, (3, 3), activation="relu", padding=1),
        layers.Convolution(128, (3, 3), activation="relu", padding=1),
        layers.MaxPooling((2, 2), stride=2),
        layers.Dropout(0.2),
        layers.Convolution(256, (3, 3), activation="relu", padding=1),
        layers.Convolution(256, (3, 3), activation="relu", padding=1),
        layers.MaxPooling((2, 2), stride=2),
        layers.Dropout(0.35),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),

        layers.Dropout(0.5),
        layers.Dense(10, activation="crossentropy_softmax")
    )

    # cifar_model = model.Model.load_from("Models/cifar_conv_epoch_30")

    cifar_model.compile(loss="softmax_crossentropy", optimizer="adam")

    print("Param Count:", cifar_model.get_param_count())

    cifar_model.fit(x_train, y_train_oh, epochs=200, learning_rate=0.001, lr_function=warmup_cosine_lr, augmentation_function=cifar_augmentation_chw_mx, batch_size=64, verbose=2, previously_trained_epochs=30, save_after_num_epochs=10, model_save_path="Models/cifar_conv", save_metrics=True)
    # print("Accuracy on test dataset: " + str(cifar_model.test(x_test, y_test_oh)))

    cifar_model.save_as("Models/cifar_conv_final")

if __name__ == "__main__":
    main()