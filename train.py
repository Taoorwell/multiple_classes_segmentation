from dataloader import load_image, load_mask
from models import jacard_coef, multi_unet_model, dice_loss, combined_loss
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Datasets preparation
    root_directory = r'../dataset/'
    image_datasets = load_image(root_directory, patch_size=256)
    mask_datasets = load_mask(root_directory, patch_size=256)
    # print(image_datasets.shape, mask_datasets.shape)
    X_train, X_test, y_train, y_test = train_test_split(image_datasets,
                                                        mask_datasets,
                                                        test_size=0.20,
                                                        random_state=42)
    # Model preparation
    model = multi_unet_model(n_classes=6, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)
    model.summary()
    model.compile(optimizer='adam', loss=combined_loss, metrics=[jacard_coef])
    model.fit(X_train, y_train,
              batch_size=10, verbose=1,
              epochs=10,
              validation_data=(X_test, y_test))


