import math
import numpy as np
import tensorflow as tf
import albumentations as a
from tensorflow.keras.utils import to_categorical
from dataloader import load_image, load_mask
from models import jacard_coef, multi_unet_model, dice_loss, combined_loss
from sklearn.model_selection import train_test_split


def datasets(x_, y_, batch_size, mode):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((x_, y_))

    @tf.autograph.experimental.do_not_convert
    def augment_function(x, y):
        def augment(x, y):
            aug = a.Compose([a.VerticalFlip(p=.5),
                             a.HorizontalFlip(p=.5),
                             a.RandomRotate90(p=0.5)])
            augmented = aug(image=x, mask=y)
            x2, y2 = augmented['image'], augmented['mask']
            return x2, y2
        x3, y3 = tf.numpy_function(augment, inp=(x, y), Tout=[tf.float32, tf.float32])
        x3.set_shape((256, 256) + (3,))
        y3.set_shape((256, 256) + (6,))
        return x3, y3
    if mode == 'train':
        dataset = dataset.map(augment_function, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    dataset = dataset.with_options(options)
    return dataset


if __name__ == '__main__':
    initial_learning_rate = 0.0001
    epochs = 100
    batch_size = 10
    # Datasets preparation
    root_directory = r'../dataset/'
    image_datasets = load_image(root_directory, patch_size=256)
    mask_datasets = load_mask(root_directory, patch_size=256)
    # one hot for mask datasets
    labels = np.array(mask_datasets)
    labels = np.expand_dims(labels, axis=3)
    print("Unique labels in label dataset are: ", np.unique(labels))
    mask_datasets = to_categorical(labels, num_classes=len(np.unique(labels)))
    # print(image_datasets.shape, mask_datasets.shape)
    X_train, X_test, y_train, y_test = train_test_split(image_datasets, mask_datasets, test_size=0.20, random_state=42)

    train_datasets = datasets(X_train, y_train, batch_size=10, mode='train')
    valid_datasets = datasets(X_test[:-1], y_test[:-1], batch_size=10, mode='valid')
    # Model preparation
    optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)

    def lr_cosine_decay(e):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * e / epochs))
        return initial_learning_rate * cosine_decay

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = multi_unet_model(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)
        model.compile(optimizer=optimizer, loss=combined_loss, metrics=[jacard_coef])
    model.summary()
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=1)
    model.fit(train_datasets,
              steps_per_epoch=len(train_datasets),
              verbose=1,
              epochs=epochs,
              validation_data=valid_datasets,
              validation_steps=len(valid_datasets),
              callbacks=[learning_rate_scheduler])

    # model save
    model.save_weights('../weights/ckpt_no_weight')


