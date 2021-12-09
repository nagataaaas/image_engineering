import math

import numpy as np
from PIL import Image
from scipy.fft import dct, idct


def read_img(filename: str) -> np.ndarray:
    return np.asarray(Image.open(filename))


def save_img(image: np.ndarray, filename: str):
    img = Image.fromarray(image)
    img.save(filename)


def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def psnr(original: np.ndarray, edited: np.ndarray) -> float:
    mse = np.mean((original.astype(float) - edited.astype(float)) ** 2)
    return 10 * np.log10((255 ** 2) / mse)


def dct_each_block(array: np.ndarray, dct_callback: callable, size=8) -> np.ndarray:
    if len(array.shape) != 2:
        raise ValueError(f'array is not a 2-Dimensional array. given shape: {array.shape}')
    height, width = array.shape
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError(f'array is not a valid 2-Dimensional array. given shape: {array.shape}')
    if height % size or width % size:
        raise ValueError(f'array is not dividable into {size}-sized block. given shape: {array.shape}')

    new = np.zeros(array.shape)
    for y in range(height // size):
        for x in range(width // size):
            dct_val = dct2(array[y * size:(y + 1) * size, x * size:(x + 1) * size])
            dct_val = dct_callback(dct_val)
            new[y * size:(y + 1) * size, x * size:(x + 1) * size] = dct_val
    return new


def idct_each_block(array: np.ndarray, size=8) -> np.ndarray:
    height, width = array.shape
    new = np.zeros(array.shape)

    for y in range(height // size):
        for x in range(width // size):
            new[y * size:(y + 1) * size, x * size:(x + 1) * size] = \
                idct2(array[y * size:(y + 1) * size, x * size:(x + 1) * size])
    return new.round().astype(np.uint8)


def callback_1(array: np.ndarray) -> np.ndarray:
    return np.rot90(np.tril(np.rot90(array)), 3)


def callback_2(minimum: int) -> callable:
    def wrapper(array: np.ndarray) -> np.ndarray:
        new = array.copy()
        new[np.abs(new) <= minimum] = 0
        return new

    return wrapper


def callback_3(ratio: float) -> callable:
    def wrapper(array: np.ndarray) -> np.ndarray:
        flat = np.abs(array.flatten())
        flat.sort()
        min_index = math.ceil(len(flat) * ratio) - 1
        if min_index == len(flat):
            min_index -= 1
        minimum = flat[min_index]

        new = array.copy()
        new[np.abs(new) <= minimum] = 0
        return new

    return wrapper


def callback_4(ratio: float) -> callable:
    def wrapper(array_: np.ndarray) -> np.ndarray:
        array = array_.copy()
        size, _ = array.shape
        index = 0
        from_index = array.size * (1 - ratio)
        for trial in range(size * 2 - 1):
            y = trial
            y = min(y, size - 1)
            for offset in range(trial - y, min(y, size) + 1):
                y_ = y - offset + max(0, trial - size + 1)
                if trial % 2:
                    y_, offset = offset, y_
                if index >= from_index:
                    array[y_, offset] = 0
                index += 1
        return array

    return wrapper


def callback_5(q: float, is_dct=True) -> callable:
    T = np.asarray([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])
    if not 0 <= q < 100:
        raise ValueError(f'{q} is not valid for argument `q`. 0 <= q < 100 is only acceptable.')
    if q < 50:
        Q = T * 50 / q
    else:
        Q = T * (100 - q) / 50

    def wrapper(array: np.ndarray) -> np.ndarray:
        if is_dct:
            return np.round(array / Q)
        return np.round(array * Q)

    return wrapper


def idct_each_block_quiz5(array: np.ndarray, q: float, size=8) -> np.ndarray:
    height, width = array.shape
    new = np.zeros(array.shape)

    for y in range(height // size):
        for x in range(width // size):
            new[y * size:(y + 1) * size, x * size:(x + 1) * size] = \
                idct2(callback_5(q, False)(array[y * size:(y + 1) * size, x * size:(x + 1) * size]))
    return new.round().astype(np.uint8)


if __name__ == '__main__':
    lena = read_img('LENA.png')

    quiz1 = dct_each_block(lena, callback_1)
    i_quiz1 = idct_each_block(quiz1)
    save_img(i_quiz1, 'quiz1.png')
    print(psnr(lena, i_quiz1))

    quiz2 = dct_each_block(lena, callback_2(10))
    i_quiz2 = idct_each_block(quiz2)
    save_img(i_quiz2, 'quiz2.png')
    print(psnr(lena, i_quiz2))

    quiz3 = dct_each_block(lena, callback_3(0.6))
    i_quiz3 = idct_each_block(quiz3)
    save_img(i_quiz3, 'quiz3.png')
    print(psnr(lena, i_quiz3))

    quiz4 = dct_each_block(lena, callback_4(0.6))
    i_quiz4 = idct_each_block(quiz4)
    save_img(i_quiz4, 'quiz4.png')
    print(psnr(lena, i_quiz4))

    q = 30
    quiz5 = dct_each_block(lena, callback_5(q))
    i_quiz5 = idct_each_block_quiz5(quiz5, q)
    save_img(i_quiz5, 'quiz5.png')
    print(psnr(lena, i_quiz5))

    # show_image(i_quiz1)
    # show_image(i_quiz2)
    # show_image(i_quiz3)
    # show_image(i_quiz4)
    # show_image(i_quiz5)
