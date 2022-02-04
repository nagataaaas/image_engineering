import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_img(filename: str) -> np.ndarray:
    img = np.asarray(Image.open(filename), dtype=np.int32)
    return img


def diff_img(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return img2 - img1


def show_img(img: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.show()


def show_hist(img: np.ndarray):
    plt.hist(img.ravel(), bins=511, range=(-255.0, 255.0), fc='k', ec='k')
    plt.xticks([-255, -128, 0, 128, 255])
    plt.savefig('hist.png')


if __name__ == '__main__':
    image_to_diff = (1, 2)
    img1, img2 = read_img(f'motion/motion{image_to_diff[0]:0>2}_512.png'), \
                 read_img(f'motion/motion{image_to_diff[1]:0>2}_512.png')
    diff = diff_img(img1, img2)
    print(sorted(list(set(diff.flatten()))))
    show_hist(diff)
    show_img(np.abs(diff))
