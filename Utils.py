import cv2
import numpy as np

from Define import *

def one_hot(cls_index, classes):
    vector = np.zeros((classes), dtype = np.float32)
    vector[cls_index] = 1.
    return vector

def Save(fake_images, save_path):
    # 10 x 10
    # 280 x 280
    save_image = np.zeros((IMAGE_HEIGHT * SAVE_HEIGHT, IMAGE_WIDTH * SAVE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)

    # 0 ~ 255
    # -1 ~ 1 -> 0 ~ 2 * 127,5 -> 0 ~ 255
    for y in range(SAVE_HEIGHT):
        for x in range(SAVE_WIDTH):
            fake_image = (fake_images[y * SAVE_WIDTH + x] + 1) * 127.5 # -1 ~ 1 (tanh)
            fake_image = fake_image.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 1)).astype(np.uint8)

            save_image[y * IMAGE_HEIGHT : (y + 1) * IMAGE_HEIGHT, x * IMAGE_WIDTH : (x + 1) * IMAGE_WIDTH] = fake_image

    cv2.imwrite(save_path, save_image)