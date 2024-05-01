# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
from osgeo import gdal
import cv2
import matplotlib.pyplot as plt
import random

batch_size = 1
num_per_pack = 10       # Number of images per pack
num_bands = 4     # Number of bands (channels) per image

# downsampling step in the UNet
def downsample(filters, size):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                      activation=tf.nn.relu,
                                      use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                      activation=tf.nn.relu,
                                      use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      activation=tf.nn.relu,
                                      use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())

    return result


# upsampling step in the UNet
def upsample(filters, size):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',
                                        activation=tf.nn.relu,
                                        use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                      activation=tf.nn.relu,
                                      use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                      activation=tf.nn.relu,
                                      use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())

    return result


# U-Net model with two heads, one for cloud masking, another for cloud brightness
def unet(num_bands = num_bands):
    down_stack = [
        downsample(16, 3),
        downsample(32, 3),
        downsample(64, 3),
    ]

    up_stack = [
        upsample(32, 3),
        upsample(16, 3),
    ]

    # cloud mask head
    last = tf.keras.Sequential()
    last.add(tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same',
                                            activation=tf.nn.relu))  # (bs, 128, 128, 16)
    last.add(tf.keras.layers.BatchNormalization())

    last.add(tf.keras.layers.Conv2D(16, 3, strides=1, padding='same',
                                    activation=tf.nn.relu))
    last.add(tf.keras.layers.BatchNormalization())

    last.add(tf.keras.layers.Conv2D(1, 1, strides=1, padding='same',
                                    activation=tf.sigmoid))

    # cloud brightness head
    last2 = tf.keras.Sequential()
    last2.add(tf.keras.layers.Conv2D(16, 1, strides=1, padding='same',
                                            activation=tf.nn.relu))
    last2.add(tf.keras.layers.BatchNormalization())

    last2.add(tf.keras.layers.Conv2D(16, 1, strides=1, padding='same',
                                    activation=tf.nn.relu))
    last2.add(tf.keras.layers.BatchNormalization())

    last2.add(tf.keras.layers.Conv2D(4, 1, strides=1, padding='same',
                                    activation=tf.sigmoid))


    skips = []
    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, None])

    bs = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]

    inputs_ = tf.reshape(inputs, [bs, height, width, -1, num_bands])
    inputs_ = tf.transpose(inputs_, [0, 3, 1, 2, 4])
    x = tf.reshape(inputs_, [-1, height, width, num_bands])

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    mask = last(x) # cloud mask

    x = tf.reduce_max(x, [1, 2], keepdims=True)
    glight = last2(x) # cloud brightness

    print(mask.shape, glight.shape)

    return tf.keras.Model(inputs=inputs, outputs=[mask, glight])

u_model_test = unet()
u_model_test.load_weights('models/u_model_autocm')

def tif2array(input_file):
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                     dtype=int)

    for b in range(dataset.RasterCount):
        band = dataset.GetRasterBand(b + 1)
        image[:, :, b] = band.ReadAsArray()

    return image

# image stretch preprocessing
def img_stretch(img, cut=2):
    min_percent = cut   # Low percentile
    max_percent = 100-cut  # High percentile
    lo, hi = np.percentile(img[img>0], (min_percent, max_percent))
    res_img = (img - lo) / (hi-lo)
    res_img[res_img<0] = 0
    res_img[res_img>1] = 1
    return res_img

# img: a planetscope image tile that has 4 bands
def autocm_test(img, is_stretch=True, vis=True):
    # image stretch preprocessing
    if is_stretch:
        img = img_stretch(img) # recommended
    else:
        img = img/10000

    height, width, num_bands = tf.shape(img)
    img = tf.reshape(img, [1, height, width, num_bands])

    if height%8 != 0:
        extend_height = (height//8 + 1)*8
    else:
        extend_height = height

    if width%8 != 0:
        extend_width = (width//8 + 1)*8
    else:
        extend_width = width
    img = tf.image.resize_with_crop_or_pad(img, extend_height, extend_width)

    pred, _ = u_model_test(img, training=False)
    pred = tf.image.resize_with_crop_or_pad(pred, height, width)

    if vis:
        plt.figure(figsize=(10, 5))

        # Image and prediction are resampled to accelerate the display.
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(img[0, ::4, ::4, 2::-1], vmin=0, vmax=1)

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(pred[0, ::4, ::4, 0], vmin=0, vmax=1)

    return pred.numpy()

# test image dataset
data = '/content/drive/MyDrive/cloud_mapping_planet_test/dataset'
imgs = [os.path.join(data, f) for f in os.listdir(data) if f.endswith('.tif')]
list.sort(imgs)

for img_path in imgs:
    # Test image size can be different from the training patch size
    img = tif2array(img_path)
    pred = autocm_test(img, is_stretch=True, vis=True)
