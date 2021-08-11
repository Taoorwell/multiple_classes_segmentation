import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from dataloader import load_image, load_mask
from models import jacard_coef, multi_unet_model, dice_loss, combined_loss
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
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
    # Model preparation
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model = multi_unet_model(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)
    model.summary()
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[jacard_coef])
    model.fit(X_train, y_train,
              batch_size=10, verbose=1,
              epochs=100,
              validation_data=(X_test, y_test))


