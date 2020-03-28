from __future__ import print_function

import os
import numpy as np
from skimage.io import imsave, imread


data_path = 'data/'

image_rows = 720
image_cols = 1280


def create_train_data():
    train_blur_data_path = os.path.join(data_path, 'train/blur/')
    train_sharp_data_path = os.path.join(data_path, 'train/sharp/')
    images_blur = os.listdir(train_blur_data_path)
    images_sharp = os.listdir(train_sharp_data_path)
    total_blur = len(images_blur)
    total_sharp = len(images_sharp)

    imgs_blur = np.ndarray((total_blur, image_rows, image_cols,3), dtype=np.uint8)
    imgs_sharp = np.ndarray((total_sharp, image_rows, image_cols,3), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creating  train images...')
    print('-' * 30)
    for image_name in images_blur:
        img_blur = imread(os.path.join(train_blur_data_path, image_name))

        img_blur = np.array([img_blur])

        imgs_blur[i] = img_blur

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total_blur))
        i += 1
    print('Loading done.')
    j=0
    for image_name in images_sharp:
        img_sharp = imread(os.path.join(train_sharp_data_path, image_name))

        img_sharp = np.array([img_sharp])

        imgs_sharp[j] = img_sharp

        if j % 100 == 0:
            print('Done: {0}/{1} images'.format(j, total_sharp))
        j += 1
    print('Loading done.')

    np.save('imgs_blur_train.npy', imgs_blur)
    np.save('imgs_sharp_train.npy', imgs_sharp)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_blur_train = np.load('imgs_blur_train.npy')
    imgs_sharp_train = np.load('imgs_sharp_train.npy')
    return imgs_blur_train, imgs_sharp_train

def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name))

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    #create_train_data()
    create_test_data()