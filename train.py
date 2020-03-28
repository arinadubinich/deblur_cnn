import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from model import modelsClass

from data import load_train_data, load_test_data

img_rows = 720
img_cols = 1280

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_blur_train, imgs_sharp_train = load_train_data()

    imgs_blur_train = preprocess(imgs_blur_train)
    imgs_sharp_train = preprocess(imgs_sharp_train)

    imgs_blur_train = imgs_blur_train.astype('float32')
    imgs_blur_train /= 255.  # scale  to [0, 1]

    imgs_sharp_train = imgs_sharp_train.astype('float32')
    imgs_sharp_train /= 255.  # scale  to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    models = modelsClass(img_rows,img_cols)
    model = models.getDeepBlind()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_blur_train, imgs_sharp_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255.  # scale  to [0, 1]

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting  on sharp images test data...')
    print('-'*30)
    imgs_sharp_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_sharp_test.npy', imgs_sharp_test)

    print('-' * 30)
    print('Saving predicted sharp images to files...')
    print('-' * 30)
    pred_dir = 'output'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_sharp_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_sharp.png'), image)

if __name__ == '__main__':
    train_and_predict()