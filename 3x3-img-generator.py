import numpy as np
from PIL import Image


def make_images(n=1):
    for i in range(n):
        arr = np.random.randint(2, size=(3, 3), dtype=np.uint8)
        arr[arr > 0] = 255  # change all ones to white color
        img = Image.fromarray(arr)
        img.save('img-{0}.jpg'.format(i))

make_images()

# TODO: create images and arrays correspond them
