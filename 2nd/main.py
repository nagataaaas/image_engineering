import numpy as np
from PIL import Image
from matplotlib import pyplot
import japanize_matplotlib


def read_img(filename: str) -> np.ndarray:
    return np.asarray(Image.open(filename))


def show_image_hist(image: np.ndarray, title: str, option: dict):
    pyplot.hist(image.flatten(), **option)
    pyplot.title(title)
    pyplot.savefig(title + '.png')
    pyplot.show()
    pyplot.close()
    print('分散:', round(image.var(), 1))


def a(image: np.ndarray) -> np.ndarray:
    new_image = image.copy().astype(np.int16)
    new_image[:, 1:] -= image[:, :-1]
    return new_image


def b(image: np.ndarray) -> np.ndarray:
    new_image = image.copy().astype(np.int16)
    new_image[1:, 1:] -= (new_image[1:, :-1] + new_image[:-1, 1:] - new_image[:-1, :-1])
    return new_image


def c(image: np.ndarray) -> np.ndarray:
    new_image = image.copy().astype(np.int16)
    # round前に0.1足すことで、0.5を1に切り上げるようにする
    new_image[1:, 1:] -= np.round((new_image[1:, :-1] + new_image[:-1, 1:]) / 2 + 0.1).astype(np.int16)
    return new_image


if __name__ == '__main__':
    img = read_img('LENA_GRAY.png')
    a_img = a(img)
    b_img = b(img)
    c_img = c(img)
    show_image_hist(img, '元画像のヒストグラム', {'bins': 256, 'range': (0, 255)})
    show_image_hist(a_img, 'Aのヒストグラム', {'bins': 512, 'range': (-256, 255)})
    show_image_hist(b_img, 'Bのヒストグラム', {'bins': 512, 'range': (-256, 255)})
    show_image_hist(c_img, 'Cのヒストグラム', {'bins': 512, 'range': (-256, 255)})
