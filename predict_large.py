import time
from sys import stdout
import cv2 as cv
import numpy as np
from dataloader import label_2d_to_rgb, scaler
from models import multi_unet_model
from matplotlib import pyplot as plt


def compute_pyramid_patch_weight_loss(width, height):
    """Compute a weight matrix that assigns bigger weight on pixels in
center and
    less weight to pixels on image boundary.
    This weight matrix is then used for merging individual tile
predictions and helps dealing
    with prediction artifacts on tile boundaries.

    Taken from & credit to:

https://github.com/BloodAxe/pytorch-toolbelt/blob/f3acfca5da05cd7ccdd85e8d343d75fa40fb44d9/pytorch_toolbelt/inference/tiles.py#L16-L50

    Args:
        width: Tile width
        height: Tile height
    Returns:
        The weight mask as ndarray
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W


def predict_on_array(model,
                     arr,
                     in_shape,
                     out_bands,
                     stride=None,
                     drop_border=0,
                     batchsize=64,
                     dtype="float32",
                     augmentation=False,
                     no_data=None,
                     verbose=False,
                     report_time=False):
    """
    Applies a pytorch segmentation model to an array in a strided
manner.

    Call model.eval() before use!

    Args:
        model: pytorch model - make sure to call model.eval() before
using this function!
        arr: HWC array for which the segmentation should be created
        stride: stride with which the model should be applied. Default:
output size
        batchsize: number of images to process in parallel
        dtype: desired output type (default: float32)
        augmentation: whether to average over rotations and mirrorings
of the image or not. triples computation time.
        no_data: a no-data vector. its length must match the number of
layers in the input array.
        verbose: whether or not to display progress
        report_time: if true, returns (result, execution time)

    Returns:
        An array containing the segmentation.
    """
    t0 = None

    if augmentation:
        operations = (lambda x: x,
                      lambda x: np.rot90(x, 1),
                      # lambda x: np.rot90(x, 2),
                      # lambda x: np.rot90(x, 3),
                      # lambda x: np.flip(x,0),
                      lambda x: np.flip(x, 1))

        inverse = (lambda x: x,
                   lambda x: np.rot90(x, -1),
                   # lambda x: np.rot90(x, -2),
                   # lambda x: np.rot90(x, -3),
                   # lambda x: np.flip(x,0),
                   lambda x: np.flip(x, 1))
    else:
        operations = (lambda x: x,)
        inverse = (lambda x: x,)

    assert in_shape[0] == in_shape[1], "Input shape must be equal in first two dims."
    out_shape = (in_shape[0] - 2 * drop_border, in_shape[1] - 2 * drop_border, out_bands)
    in_size = in_shape[0]
    out_size = out_shape[0]
    stride = stride or out_size
    pad = (in_size - out_size)//2
    assert pad % 2 == 0, "Model input and output shapes have to be divisible by 2."

    weight_mask = compute_pyramid_patch_weight_loss(out_size, out_size)

    original_size = arr.shape
    ymin = 0
    xmin = 0

    if no_data is not None:
        # assert arr.shape[-1]==len(no_data_vec), "Length of no_data_vec must match number of channels."
        # data_mask = np.all(arr[:,:,0].reshape( (-1,arr.shape[-1]) ) != no_data, axis=1).reshape(arr.shape[:2])
        nonzero = np.nonzero(arr[:, :, 0]-no_data)
        ymin = np.min(nonzero[0])
        ymax = np.max(nonzero[0])
        xmin = np.min(nonzero[1])
        xmax = np.max(nonzero[1])
        img = arr[ymin:ymax, xmin:xmax]

    else:
        img = arr

    final_output = np.zeros(img.shape[:2]+(out_shape[-1],), dtype=dtype)

    op_cnt = 0
    for op, inv in zip(operations, inverse):
        img = op(img)
        img_shape = img.shape
        x_tiles = int(np.ceil(img.shape[1]/stride))
        y_tiles = int(np.ceil(img.shape[0]/stride))

        y_range = range(0, (y_tiles+1)*stride-out_size, stride)
        x_range = range(0, (x_tiles+1)*stride-out_size, stride)

        y_pad_after = y_range[-1]+in_size-img.shape[0]-pad
        x_pad_after = x_range[-1]+in_size-img.shape[1]-pad

        output = np.zeros((img.shape[0]+y_pad_after-pad, img.shape[1]+x_pad_after-pad)+(out_shape[-1],), dtype=dtype)
        division_mask = np.zeros(output.shape[:2], dtype=dtype) + 1E-7
        img = np.pad(img, ((pad, y_pad_after), (pad, x_pad_after), (0, 0)), mode='reflect')

        patches = len(y_range)*len(x_range)

        def patch_generator():
            for y in y_range:
                for x in x_range:
                    yield img[y:y+in_size, x:x+in_size]

        patch_gen = patch_generator()

        y = 0
        x = 0
        patch_idx = 0
        batchsize_ = batchsize

        t0 = time.time()

        while patch_idx < patches:
            batchsize_ = min(batchsize_, patches, patches - patch_idx)
            patch_idx += batchsize_
            if verbose:
                stdout.write("\r%.2f%%" % (100 * (patch_idx + op_cnt * patches) / (len(operations) * patches)))

            batch = np.zeros((batchsize_,) + in_shape, dtype=dtype)

            for j in range(batchsize_):
                batch[j] = next(patch_gen)

            # EDIT HERE
            prediction = model.predict(batch)
            if drop_border > 0:
                prediction = prediction[:, drop_border:-drop_border, drop_border:-drop_border, :]

            for j in range(batchsize_):
                output[y:y + out_size, x:x + out_size] += prediction[j] * weight_mask[..., None]
                division_mask[y:y + out_size, x:x + out_size] += weight_mask
                x += stride
                if x + out_size > output.shape[1]:
                    x = 0
                    y += stride

        output = output / division_mask[..., None]
        output = inv(output[:img_shape[0], :img_shape[1]])
        final_output += output
        img = arr[ymin:ymax, xmin:xmax] if no_data is not None else arr
        op_cnt += 1
        if verbose:
            stdout.write("\rAugmentation step %d/%d done.\n" % (op_cnt, len(operations)))

    if verbose:
        stdout.flush()

    final_output = final_output/len(operations)

    if no_data is not None:
        final_output = np.pad(final_output, ((ymin, original_size[0] - ymax), (xmin, original_size[1]-xmax), (0, 0)),
                              mode='constant', constant_values=0)

    if report_time:
        return final_output, time.time() - t0

    else:
        return final_output


if __name__ == '__main__':

    # Read the image needed to be predicted
    patch_size = 256
    file_path = r'../dataset/Tile 1/images/image_part_008.jpg'
    mask_path = r'../dataset/Tile 1/masks/image_part_008.png'

    image = cv.imread(file_path, 1)
    mask = cv.imread(mask_path, 1)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    print(image.shape, mask.shape)

    image = scaler.fit_transform(
        image.reshape(-1, image.shape[-1])).reshape(image.shape)
    print(image.shape)

    # load model
    model = multi_unet_model(n_classes=6, IMG_WIDTH=patch_size, IMG_HEIGHT=patch_size, IMG_CHANNELS=3)
    # model.summary()
    model.load_weights(r'../weights/ckpt_no_weight')
    #
    # # model prediction
    pre, _ = predict_on_array(model=model,
                              arr=image,
                              in_shape=(patch_size, patch_size, 3),
                              out_bands=6,
                              stride=250,
                              batchsize=4,
                              augmentation=True,
                              verbose=True,
                              report_time=True)

    pre1 = np.argmax(pre, axis=-1)
    print(pre1.shape)
    pre_rgb = label_2d_to_rgb(pre1)
    plt.subplot(131)
    plt.imshow(image)

    plt.subplot(132)
    plt.imshow(mask)

    plt.subplot(133)
    plt.imshow(pre_rgb)
    plt.show()

