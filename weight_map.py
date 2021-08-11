import tensorflow as tf
import numpy as np
import cv2 as cv
from dataloader import rgb_to_2d_label
from matplotlib import pyplot as plt


def log_conv(y_true):
    y_true = tf.math.argmax(y_true, axis=-1)
    # Laplacian for edge extraction
    laplacian_filter = tf.constant([[0, .25, 0], [.25, -1, .25], [0, .25, 0]],
                                   dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, (3, 3, 1, 1))

    output = tf.nn.conv2d(y_true, filters=laplacian_filter, strides=1, padding=[[0, 0], [0, 0], [0, 0], [0, 0]])

    edge = output != 0
    edge = tf.cast(edge, tf.float32)

    # Gaussian blur for pixel weight
    def gaussian_2d(ksize, sigma=1):
        m = (ksize - 1) / 2
        y, x = np.ogrid[-m:m+1, -m:m+1]
        value = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        # value[value < np.finfo(value.dtype).eps * value.max()] = 0
        sum_v = value.sum()
        if sum_v != 0:
            value /= sum_v
        return value

    gaussian_filter = gaussian_2d(ksize=3, sigma=1)
    gaussian_filter = np.reshape(gaussian_filter, (3, 3, 1, 1))

    pixel_weight = 10 * tf.nn.conv2d(edge, filters=gaussian_filter, strides=1, padding='SAME') + 1

    return pixel_weight


if __name__ == '__main__':
    mask_path = r'../dataset/Tile 1/masks/image_part_002.png'
    mask = cv.imread(mask_path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    mask = rgb_to_2d_label(mask)
    mask = np.expand_dims(mask, axis=(0, -1))
    edge, pixel_weight = log_conv(mask)

    # print(pixel_weight)
    plt.imshow(pixel_weight[0, :, :, 0])
    plt.show()
