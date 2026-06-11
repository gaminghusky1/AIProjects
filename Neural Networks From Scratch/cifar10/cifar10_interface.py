import sys
import pygame
import mlx.core as mx
import numpy as np
from keras.datasets import cifar10
from mlx_model import *

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = mx.array(x_train)
    # y_train = mx.array(y_train)
    # x_test = mx.array(x_test)
    # y_test = mx.array(y_test)

    x_train = np.transpose(x_train, (0, 3, 2, 1))
    x_test = np.transpose(x_test, (0, 3, 2, 1))

    # x_train = x_train.reshape(50000, 3072)
    # x_test = x_test.reshape(10000, 3072)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def get_image_surface(img, scale=1):
    # Ensure shape is (H, W, C)
    if img.shape == (3, 32, 32):  # CIFAR format
        img = np.transpose(img, (1, 2, 0))

    # Ensure uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # Create surface
    surface = pygame.surfarray.make_surface(img)

    # Scale if needed
    if scale != 1:
        surface = pygame.transform.scale(
            surface,
            (img.shape[1] * scale, img.shape[0] * scale)
        )

    return surface

def get_smooth_image_surface(img, scale=1):
    # Ensure shape is (H, W, C)
    if img.shape == (3, 32, 32):  # CIFAR format
        img = np.transpose(img, (1, 2, 0))

    # Ensure uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # Create surface
    surface = pygame.surfarray.make_surface(img)

    # Scale if needed
    if scale != 1:
        surface = pygame.transform.smoothscale(
            surface,
            (img.shape[1] * scale, img.shape[0] * scale)
        )

    return surface


def main():
    # Colors
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (28, 128, 28)
    yellow = (230, 230, 0)
    brown = (118, 92, 72)
    gray = (175, 175, 175)
    dark_gray = (99, 102, 106)
    blue = (12, 246, 242)
    aqua = (5, 195, 221)
    red = (255, 0, 0)

    x_train, y_train, x_test, y_test = load_cifar10()

    pygame.init()
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.QUIT, pygame.KEYUP, pygame.MOUSEBUTTONDOWN])

    monitor_width = pygame.display.get_desktop_sizes()[0][0]
    scale = monitor_width / 80 - 2
    # scale = 30

    screen = pygame.display.set_mode((80 * scale, 32 * scale))
    pygame.display.set_caption('CIFAR')

    font = pygame.font.SysFont('Arial', 48)
    small_font = pygame.font.SysFont('Arial', 20)

    screen_width = screen.get_width()
    screen_height = screen.get_height()

    img_idx = 0
    curr_x = x_train
    curr_y = y_train

    ordered_labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    cifar_model = model.Model.load_from("Models/cifar_conv_epoch_50")
    # Main Game Loop
    while True:
        screen.fill(black)

        screen.blit(get_image_surface(curr_x[img_idx], scale), (0, 0))
        screen.blit(get_smooth_image_surface(curr_x[img_idx], scale), (48 * scale, 0))

        curr_dataset = "train" if curr_x is x_train else "test"
        curr_dataset += f" ({img_idx + 1} of {len(curr_x)})"
        curr_dataset += f"; true label: {ordered_labels[curr_y[img_idx]]}"

        dataset_surface = small_font.render(curr_dataset, True, white)
        dataset_rect = dataset_surface.get_rect(center=(scale * (32 + (48 - 32) // 2), 1 * scale))
        screen.blit(dataset_surface, dataset_rect)

        # Kinda bad b/w numpy and mlx
        # Convert to mlx.core.array so the model can take it in, also add newaxis for batch dimension
        x = mx.array(curr_x[img_idx][np.newaxis, ...], dtype=mx.float32)
        # Get model predictions, take [0] to remove batch dimension, * 100 for percentages
        y = np.array(cifar_model.predict(x)[0] * 100)

        # Add integer labels to each prediction percentage
        predictions = np.column_stack((np.arange(len(y)), y))
        # Sort predictions by percentage
        predictions = predictions[np.argsort(-predictions[:, 1])]

        for i in range(len(predictions)):
            label = int(predictions[i][0])
            confidence = predictions[i][1]
            label_surface = font.render(f"{ordered_labels[label]}: {confidence:.2f}%", True, green if curr_y[img_idx] == label else white)
            label_rect = label_surface.get_rect(center=(scale * (32 + (48 - 32) // 2), 3 * (i + 1) * scale))
            screen.blit(label_surface, label_rect)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    img_idx = (img_idx - 1) % curr_x.shape[0]
                elif event.key == pygame.K_RIGHT:
                    img_idx = (img_idx + 1) % curr_x.shape[0]
                elif event.key == pygame.K_LEFTBRACKET:
                    curr_x = x_train
                    curr_y = y_train
                    img_idx = 0
                elif event.key == pygame.K_RIGHTBRACKET:
                    curr_x = x_test
                    curr_y = y_test
                    img_idx = 0
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()


if __name__ == "__main__":
    main()