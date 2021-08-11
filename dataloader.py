import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from tensorflow.keras.utils import to_categorical

scaler = MinMaxScaler()


def load_image(root_directory, patch_size=256):
    image_dataset = []
    for path, subdirs, files in os.walk(root_directory):
        # print(path)
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':  # Find all 'images' directories
            images = sorted(os.listdir(path))  # List of all image names in this subdirectory
            for i, image_name in enumerate(images):
                if image_name.endswith(".jpg"):  # Only read jpg images...

                    image = cv2.imread(path + "/" + image_name, 1)  # Read each image as BGR
                    SIZE_X = (image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
                    SIZE_Y = (image.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
                    image = Image.fromarray(image)
                    image = image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
                    # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    image = np.array(image)

                    # Extract patches from each image
                    print("Now patchifying image:", path + "/" + image_name)
                    patches_img = patchify(image, (patch_size, patch_size, 3),
                                           step=patch_size)  # Step=256 for 256 patches means no overlap
                    # print(patches_img.shape)

                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            single_patch_img = patches_img[i, j]

                            # Use min-max scaler instead of just dividing by 255.
                            single_patch_img = scaler.fit_transform(
                                single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)

                            # single_patch_img = (single_patch_img.astype('float32')) / 255.
                            single_patch_img = single_patch_img[
                                0]  # Drop the extra unecessary dimension that patchify adds.
                            image_dataset.append(single_patch_img)
    return np.array(image_dataset, dtype=np.float32)

# Now do the same as above for masks
# For this specific dataset we could have added masks to the above code as masks have extension png


def load_mask(root_directory, patch_size=256):
    mask_dataset = []
    for path, subdirs, files in os.walk(root_directory):
        # print(path)
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':  # Find all 'images' directories
            masks = sorted(os.listdir(path))  # List of all image names in this subdirectory
            for i, mask_name in enumerate(masks):
                if mask_name.endswith(".png"):  # Only read png images... (masks in this dataset)

                    mask = cv2.imread(path + "/" + mask_name,
                                      1)
                    # Read each image as Grey (or color but remember to map each color to an integer)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    SIZE_X = (mask.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
                    SIZE_Y = (mask.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
                    # mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    mask = np.array(mask)

                    # Extract patches from each image
                    print("Now patchifying mask:", path + "/" + mask_name)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3),
                                            step=patch_size)  # Step=256 for 256 patches means no overlap

                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            single_patch_mask = patches_mask[i, j]
                            # single_patch_img = (single_patch_img.astype('float32')) / 255.
                            # #No need to scale masks, but you can do it if you want
                            single_patch_mask = single_patch_mask[
                                0]  # Drop the extra unecessary dimension that patchify adds.
                            mask_dataset.append(single_patch_mask)
    mask_dataset = np.array(mask_dataset)
    labels = []
    for i in range(mask_dataset.shape[0]):
        label = rgb_to_2d_label(mask_dataset[i])
        labels.append(label)
    return labels

# Sanity check, view few mages
# image_number = np.random.randint(0, len(image_dataset))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
# plt.subplot(122)
# plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
# plt.show()

###########################################################################
"""
RGB to HEX: (Hexadecimel --> base 16)
This number divided by sixteen (integer division; ignoring any remainder) gives 
the first hexadecimal digit (between 0 and F, where the letters A to F represent 
the numbers 10 to 15). The remainder gives the second hexadecimal digit. 
0-9 --> 0-9
10-15 --> A-F
Example: RGB --> R=201, G=, B=
R = 201/16 = 12 with remainder of 9. So hex code for R is C9 (remember C=12)
Calculating RGB from HEX: #3C1098
3C = 3*16 + 12 = 60
10 = 1*16 + 0 = 16
98 = 9*16 + 8 = 152
"""
# Convert HEX to RGB array
# Try the following to understand how python handles hex values...
a = int('3C', 16)  # 3C with base 16. Should return 60.
print(a)
# Do the same for all RGB channels in each hex code to convert to RGB
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i + 2], 16) for i in (0, 2, 4)))  # 60, 16, 152

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i + 2], 16) for i in (0, 2, 4)))  # 132, 41, 246

Road = '#6EC1E4'.lstrip('#')
Road = np.array(tuple(int(Road[i:i + 2], 16) for i in (0, 2, 4)))  # 110, 193, 228

Vegetation = 'FEDD3A'.lstrip('#')
Vegetation = np.array(tuple(int(Vegetation[i:i + 2], 16) for i in (0, 2, 4)))  # 254, 221, 58

Water = 'E2A929'.lstrip('#')
Water = np.array(tuple(int(Water[i:i + 2], 16) for i in (0, 2, 4)))  # 226, 169, 41

Unlabeled = '#9B9B9B'.lstrip('#')
Unlabeled = np.array(tuple(int(Unlabeled[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

# label = single_patch_mask


# Now replace RGB to integer values to be used as labels.
# Find pixels with combination of RGB for the above defined arrays...
# if matches then replace all values in that pixel with a specific integer
def rgb_to_2d_label(label):
    """
    Suply our labale masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg[np.all(label == Building, axis=-1)] = 0
    label_seg[np.all(label == Land, axis=-1)] = 1
    label_seg[np.all(label == Road, axis=-1)] = 2
    label_seg[np.all(label == Vegetation, axis=-1)] = 3
    label_seg[np.all(label == Water, axis=-1)] = 4
    label_seg[np.all(label == Unlabeled, axis=-1)] = 5

    label_seg = label_seg[:, :, 0]  # Just take the first channel, no need for all 3 channels

    return label_seg


# Another Sanity check, view few mages
# image_number = np.random.randint(0, len(image_dataset))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(image_dataset[image_number])
# plt.subplot(122)
# plt.imshow(labels[image_number][:, :, 0])
# plt.show()
