import numpy as np
from keras.datasets import mnist
import pickle

def load_data():
    (train_images_f, train_labels_f), (test_images_f, test_labels_f) = mnist.load_data()

    # Normalize the image data to values between 0 and 1
    train_images_f = train_images_f.astype('float32') / 255
    test_images_f = test_images_f.astype('float32') / 255

    # Flatten the images from 28x28 to 784-dimensional vectors
    train_images_f = train_images_f.reshape((-1, 28, 28, 1))
    test_images_f = test_images_f.reshape((-1, 28, 28, 1))

    return train_images_f, train_labels_f, test_images_f, test_labels_f

x_train, y_train, x_test, y_test = load_data()

def get_random_images_by_class(images, labels, n_per_class, filepath="random_images.pkl", seed=42):
    np.random.seed(seed)
    selected_images = []
    selected_labels = []

    classes = np.unique(labels)
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        chosen_indices = np.random.choice(cls_indices, n_per_class, replace=False)
        selected_images.append(images[chosen_indices])
        selected_labels.append(labels[chosen_indices])

    final_images = np.concatenate(selected_images, axis=0)
    final_labels = np.concatenate(selected_labels, axis=0)

    # Shuffle while preserving correspondence
    shuffle_idx = np.random.permutation(len(final_labels))
    final_images = final_images[shuffle_idx]
    final_labels = final_labels[shuffle_idx]

    with open(filepath, "wb") as f:
        pickle.dump((final_images, final_labels), f)

    return final_images, final_labels

get_random_images_by_class(x_train, y_train, 5)
print("Saved 50 random training images to random_images.pkl")