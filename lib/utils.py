import glob
import numpy as np
from PIL import Image
import random
import skimage
import skimage.transform


def resize_imgs(imgs, width=50, height=50):
    resized_imgs = []
    for img in imgs:
        resized_img = skimage.transform.resize(img, (height, width))
        resized_imgs.append(resized_img)
    resized_imgs = np.array(resized_imgs)
    return resized_imgs


def get_mean_img(imgs):
    """Return mean image from a list of images"""
    return np.mean(imgs, axis=0)


def basic_imgs_normalization(imgs):
    """Compute and subtract the mean image from the list of images"""
    mean_img = get_mean_img(imgs)
    return imgs - mean_img


def image_liniarization(imgs):
    "Convert list of images to 1D arrays"
    return imgs.reshape((imgs.shape[0], -1))


def load_images(n_classes=10, loc="train"):
    """Load images and labels + shuffle"""
    imgs_with_labels = []
    for c in range(n_classes):
        for filename in glob.glob(f"dataset/{loc}/{c}/*"):
            img = np.array(Image.open(filename))
            img = skimage.img_as_float(img)
            imgs_with_labels.append((img, c))
    random.shuffle(imgs_with_labels)
    imgs = [x for x, _ in imgs_with_labels]
    labels = [y for _, y in imgs_with_labels]
    return imgs, labels


def load_train_images(n_classes=10):
    """Return train images and their labels, after shuffling"""
    return load_images(n_classes=n_classes, loc="train")


def load_test_images(n_classes=10):
    """Return test images and their labels, after shuffling"""
    return load_images(n_classes=n_classes, loc="test")
