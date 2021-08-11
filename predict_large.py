import cv2 as cv
import numpy as np
from PIL import Image
from dataloader import label_2d_to_rgb, scaler
from patchify import patchify
from models import multi_unet_model
import tensorflow as tf
from matplotlib import pyplot as plt

# Read the image needed to be predicted
patch_size = 256
file_path = r'../dataset/Tile 1/images/image_part_001.jpg'
mask_path = r'../dataset/Tile 1/masks/image_part_001.png'

image = cv.imread(file_path, 1)
mask = cv.imread(mask_path, 1)
mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
print(image.shape, mask.shape)

# crop image
SIZE_X = (image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
SIZE_Y = (image.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
image = Image.fromarray(image)
image = image.crop((0, 0, SIZE_X, SIZE_Y))
image = np.array(image, dtype=np.float32)


image = scaler.fit_transform(
    image.reshape(-1, image.shape[-1])).reshape(image.shape)
print(image.shape)
# patching the image into patches, preparing for prediction
# patches = patchify(image)
#
# load model
model = multi_unet_model(n_classes=6, IMG_WIDTH=SIZE_X, IMG_HEIGHT=SIZE_Y, IMG_CHANNELS=3)
model.summary()
model.load_weights(r'../weights/ckpt')
#
# # model prediction
pre = model.predict(np.expand_dims(image, axis=0))
pre = np.reshape(pre, (SIZE_X*SIZE_Y, 6))
pre1 = np.argmax(pre, axis=-1).reshape(SIZE_Y, SIZE_X)
print(pre1.shape)
pre_rgb = label_2d_to_rgb(pre1)

plt.subplot(121)
plt.imshow(mask)
plt.subplot(122)
plt.imshow(pre_rgb)
plt.show()

# for i in patches:
#     patche_pre = model.predict(i)
#     patche_pre = tf.argmax(patche_pre, axis=-1)
