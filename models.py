import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, \
    BatchNormalization, Dropout, Lambda, add, Activation


def jacard_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection + 1.0)


def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2 * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)


@tf.autograph.experimental.do_not_convert
def log_conv(y_true):
    y_true_1 = tf.math.argmax(y_true, axis=-1)
    y_true_1 = tf.cast(y_true_1, tf.float32)
    y_true_1 = tf.expand_dims(y_true_1, axis=-1)
    # Laplacian for edge extraction
    laplacian_filter = tf.constant([[0, .25, 0], [.25, -1, .25], [0, .25, 0]],
                                   dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, (3, 3, 1, 1))

    output = tf.nn.conv2d(y_true_1, filters=laplacian_filter, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])

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


def cross_entropy(y_true, y_pred, weight=False):
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                 reduction=tf.keras.losses.Reduction.NONE)
    loss = ce(y_true, y_pred)
    loss = tf.expand_dims(loss, axis=-1)
    if weight:
        pixel_weight = log_conv(y_true)
        loss = pixel_weight * loss
    return tf.reduce_mean(loss)


def combined_loss(y_true, y_pred, weight=False):
    eps = 1E-15
    ce = cross_entropy(y_true, y_pred, weight=weight)
    loss = ce - tf.math.log(jacard_coef(y_true, y_pred) + eps)
    return loss


################################################################
def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    def residual_conv_block(in_, filters):
        # main path including convolution * 2, Batch normalization * 2, Dropout and Max pooling
        out_ = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(in_)
        out_ = BatchNormalization()(out_)
        out_ = Activation(activation='relu')(out_)
        out_ = Dropout(.2)(out_)

        out_ = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(out_)
        out_ = BatchNormalization()(out_)
        out_1 = Activation(activation='relu')(out_)

        p = MaxPooling2D((2, 2))(out_1)

        # shortcut path including convolution with stride 2 and Batch normalization
        shortcut = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same',
                          strides=(2, 2))(in_)
        shortcut = BatchNormalization()(shortcut)

        # add main path and shortcut path together
        out_2 = add([p, shortcut])

        return out_1, out_2

    # Contraction path
    c1, p1 = residual_conv_block(inputs, filters=16)
    # c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    # c1 = Dropout(0.2)(c1)  # Original 0.1
    # c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    # p1 = MaxPooling2D((2, 2))(c1)

    c2, p2 = residual_conv_block(p1, filters=32)
    # c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    # c2 = Dropout(0.2)(c2)  # Original 0.1
    # c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    # p2 = MaxPooling2D((2, 2))(c2)

    c3, p3 = residual_conv_block(p2, filters=64)
    # c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    # c3 = Dropout(0.2)(c3)
    # c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    # p3 = MaxPooling2D((2, 2))(c3)

    c4, p4 = residual_conv_block(p3, filters=128)
    # c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    # c4 = Dropout(0.2)(c4)
    # c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    # p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # bridge
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    def residual_up_sampling_block(in_1, in_2, filters):
        # up sampling and concatenate
        out_ = UpSampling2D(size=(2, 2))(in_1)
        out_1 = concatenate([out_, in_2], axis=3)

        # convolution * 2, Batch normalization * 2, Dropout
        out_ = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(out_1)
        out_ = BatchNormalization()(out_)
        out_ = Activation(activation='relu')(out_)
        out_ = Dropout(.2)(out_)

        out_ = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(out_)
        out_ = BatchNormalization()(out_)
        out_ = Activation(activation='relu')(out_)

        # shortcut, Convolution and Batch normalization
        shortcut = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(out_1)
        shortcut = BatchNormalization()(shortcut)

        out_ = add([shortcut, out_])
        return out_

    # Expansive path
    c6 = residual_up_sampling_block(c5, c4, filters=128)
    # u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    # u6 = concatenate([u6, c4])
    # c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    # c6 = Dropout(0.2)(c6)
    # c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c7 = residual_up_sampling_block(c6, c3, filters=64)
    # u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    # u7 = concatenate([u7, c3])
    # c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    # c7 = Dropout(0.2)(c7)
    # c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c8 = residual_up_sampling_block(c7, c2, filters=32)
    # u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    # u8 = concatenate([u8, c2])
    # c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    # c8 = Dropout(0.2)(c8)  # Original 0.1
    # c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c9 = residual_up_sampling_block(c8, c1, filters=16)
    # u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    # u9 = concatenate([u9, c1], axis=3)
    # c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    # c9 = Dropout(0.2)(c9)  # Original 0.1
    # c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # output
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    # NOTE: Compile the model in the main program to make it easy to test with various loss functions
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    return model


if __name__ == '__main__':
    model = multi_unet_model(n_classes=6, IMG_CHANNELS=3)
    model.summary()

