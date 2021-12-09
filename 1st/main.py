import numpy as np
from PIL import Image


def simple_read_bw(original_filename: str, copy_filename: str):
    img = Image.open(original_filename)
    img.show()
    img.save(copy_filename)


def simple_color_change(original_filename: str, elem1: str, elem2: str, edit_filename: str):
    if elem1 not in 'rgb' or elem2 not in 'rgb':
        raise ValueError('{!r} or {!r} is not acceptable'.format(elem1, elem2))
    img = np.asarray(Image.open(original_filename))

    index = {'r': 0, 'g': 1, 'b': 2}
    index[elem1], index[elem2] = index[elem2], index[elem1]

    img = img[:, :, [index['r'], index['g'], index['b']]]
    img = Image.fromarray(img)
    img.save(edit_filename)


def simple_mix(first_filename: str, second_filename: str, ratio: float, edit_filename: str):
    if not 0 <= ratio <= 1:
        raise ValueError('ratio needs to be between 0 and 1. {} is invalid'.format(ratio))
    first = np.asarray(Image.open(first_filename)) * ratio
    second = np.asarray(Image.open(second_filename)) * (1 - ratio)
    img = Image.fromarray((first + second).astype(np.uint8))
    img.save(edit_filename)


if __name__ == '__main__':
    # simple_read_bw('LENA.png', 'c_LENA.png')
    simple_color_change('c_LENA.png', 'r', 'b', 'c_LENA_mod.png')
    simple_mix('Lighthouse.png', 'Cameraman.png', 0.3, 'mixed.png')
