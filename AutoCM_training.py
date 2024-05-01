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

# Input pipeline from 3 spatiotemporal packs, each with a shape of [400, 400, num_per_pack*num_bands]
def input_pipeline(filenames, batch_size, is_shuffle=True, is_train=True, is_repeat=True):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([400*400*num_per_pack*num_bands], dtype=tf.int64)
    }

    def _parse_function(example_proto):
        feature_dict = tf.io.parse_single_example(example_proto, feature_description)

        image = tf.reshape(feature_dict['image_raw'], [400, 400, num_per_pack*num_bands])
        image = tf.cast(image, tf.float32)
        image = image/10000

        return image

    def _aug(example1, example2, example3):
        image = tf.concat([example1, example2, example3], axis=-1)
        image = tf.image.rot90(image, tf.random.uniform([], 0, 5, dtype=tf.int32))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        # pack1, pack2, pack3
        return image[:, :, :num_per_pack*num_bands], \
               image[:, :, num_per_pack*num_bands:2*num_per_pack*num_bands], \
               image[:, :, 2*num_per_pack*num_bands:]

    dataset = []
    for fname in filenames:
        print(fname)
        pack = tf.data.TFRecordDataset(fname)
        pack = pack.map(_parse_function)
        dataset.append(pack)

    dataset = tf.data.Dataset.zip(tuple(dataset))
    if is_train:
        dataset = dataset.map(_aug)

    if is_repeat:
        dataset = dataset.repeat()

    if is_shuffle:
        dataset = dataset.shuffle(buffer_size=100)

    return dataset

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

def discriminator(norm_type='batchnorm'):
    inputs = tf.keras.layers.Input(shape=[None, None, 4])

    down_stack = [
        downsample(64, 3),
        downsample(32, 3),
        downsample(16, 3),
    ]
    x = inputs

    for down in down_stack:
        x = down(x)

    x = tf.reduce_mean(x, [1, 2], keepdims=True)

    last = tf.keras.layers.Conv2D(
        16, 1, strides=1,
        padding='same',
        activation=tf.nn.relu)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# Simaese network for learning spatiotemporal similarity
# input 1,2: cloud-free composite from two packs
def Siamese_net():
    input_1 = tf.keras.layers.Input(shape=[None, None, 4])
    input_2 = tf.keras.layers.Input(shape=[None, None, 4])

    d_model = discriminator()
    diff = tf.square(d_model(input_1)-d_model(input_2))

    last = tf.keras.Sequential()
    last.add(tf.keras.layers.Conv2D(
        16, 1, strides=1,
        padding='same',
        activation=tf.nn.relu))
    last.add(tf.keras.layers.BatchNormalization())

    last.add(tf.keras.layers.Conv2D(
        1, 1, strides=1,
        padding='same',
        activation=tf.sigmoid))

    final = last(diff)
    final = tf.squeeze(final, axis=[1, 2])
    return tf.keras.Model(inputs=[input_1, input_2], outputs=final)

u_class_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

u_model = unet()
d_model = Siamese_net()

# Calculate cloud-free composite
def compose(img_pack, cloud_mask):
    height = tf.shape(img_pack)[1]
    width = tf.shape(img_pack)[2]

    pack = tf.reshape(img_pack, [-1, height, width, num_per_pack, num_bands])
    pack = tf.transpose(pack, [0, 3, 1, 2, 4])

    pack_cloud = tf.reshape(1-cloud_mask, [-1, num_per_pack, height, width, 1])
    denorm = tf.reduce_sum(pack_cloud, axis=1, keepdims=True)
    composite = tf.reduce_sum(pack*pack_cloud/(denorm+1e-8), axis=1)

    return composite


def recons_loss(y_true, cloud_mask, composite, glight):
    height = tf.shape(cloud_mask)[1]
    width = tf.shape(cloud_mask)[2]

    cloud_mask = tf.reshape(cloud_mask, [-1, num_per_pack, height, width, 1])
    glight = tf.reshape(glight, [-1, num_per_pack, 1, 1, num_bands])

    composite = tf.reshape(composite, [-1, 1, height, width, num_bands])

    y_true = tf.reshape(y_true, [-1, height, width, num_per_pack, num_bands])
    y_true = tf.transpose(y_true, [0, 3, 1, 2, 4])

    recon = (1-cloud_mask)*composite + glight*cloud_mask

    loss = tf.reduce_mean(tf.square(y_true-recon))
    return loss


# Generate spatial overlapping packs
def crop_generate(train_pack):
    height = 200
    width = 200
    batch_size = tf.shape(train_pack)[0]

    x_1 = 0
    y_1 = 0
    pack_crop1 = train_pack[:, y_1:y_1+height, x_1:x_1+width, :]

    x_2 = np.array(random.choices(range(0, 201, 50), k=batch_size))
    y_2 = np.array(random.choices(range(0, 201, 50), k=batch_size))
    boxes = np.vstack([x_2, y_2, x_2+200, y_2+200]).transpose()/400
    pack_crop2 = tf.image.crop_and_resize(train_pack, boxes=boxes, box_indices=[i for i in range(batch_size)], crop_size=[200, 200])

    x_min = tf.math.maximum(x_1, x_2)
    y_min = tf.math.maximum(y_1, y_2)
    x_max = tf.math.minimum(x_1+200, x_2+200)
    y_max = tf.math.minimum(y_1+200, y_2+200)
    dx = tf.math.maximum(x_max - x_min, 0)
    dy = tf.math.maximum(y_max - y_min, 0)
    ratio = tf.cast(dx*dy, tf.float32)/40000.0

    return pack_crop1, pack_crop2, ratio


def train_step(train_pack1, train_pack2, train_pack3):
    pack1_crop1, pack1_crop2, ratio_1 = crop_generate(train_pack1)
    pack2_crop1, pack2_crop2, ratio_2 = crop_generate(train_pack2)
    pack3_crop1, pack3_crop2, ratio_3 = crop_generate(train_pack3)

    with tf.GradientTape() as gen_class_tape, tf.GradientTape() as disc_tape:
        test_feature_11, glight11 = u_model(pack1_crop1, training=True)
        test_feature_21, glight21 = u_model(pack2_crop1, training=True)

        ################### reconstruction loss ####################
        composite_11 = compose(pack1_crop1, test_feature_11)
        loss_rec_11 = recons_loss(pack1_crop1, test_feature_11, composite_11, glight11)

        composite_21 = compose(pack2_crop1, test_feature_21)
        loss_rec_21 = recons_loss(pack2_crop1, test_feature_21, composite_21, glight21)

        ################### Spatial similarity loss ####################
        test_feature_12, glight12 = u_model(pack1_crop2, training=True)
        test_feature_22, glight22 = u_model(pack2_crop2, training=True)

        composite_12 = compose(pack1_crop2, test_feature_12)
        loss_rec_12 = recons_loss(pack1_crop2, test_feature_12, composite_12, glight12)

        composite_22 = compose(pack2_crop2, test_feature_22)
        loss_rec_22 = recons_loss(pack2_crop2, test_feature_22, composite_22, glight22)

        sim_1 = d_model([composite_21, composite_12], training=True)
        sim_2 = d_model([composite_11, composite_22], training=True)

        loss_spatial_sim = tf.reduce_mean(tf.square(sim_1-ratio_1)+tf.square(sim_2-ratio_2))/2

        ################### Temporal similarity loss ####################
        test_feature_31, glight31 = u_model(pack3_crop1, training=True)
        composite_31 = compose(pack3_crop1, test_feature_31)
        loss_rec_31 = recons_loss(pack3_crop1, test_feature_31, composite_31, glight31)

        sim_1 = d_model([composite_31, composite_21], training=True)
        sim_2 = d_model([composite_31, composite_11], training=True)

        loss_temporal_sim = tf.reduce_mean(sim_2-sim_1)

        loss_rec = (loss_rec_11 + loss_rec_21 + loss_rec_12 + loss_rec_22 + loss_rec_31)/5
        loss = loss_rec+0.01*loss_spatial_sim+0.01*tf.nn.relu(loss_temporal_sim+0.2)

        gradients_of_generator_from_class = gen_class_tape.gradient(loss, u_model.trainable_variables)
        u_class_optimizer.apply_gradients(zip(gradients_of_generator_from_class, u_model.trainable_variables))

        gradients_of_disc = disc_tape.gradient(loss, d_model.trainable_variables)
        d_optimizer.apply_gradients(zip(gradients_of_disc, d_model.trainable_variables))

        return loss, loss_rec, loss_spatial_sim, loss_temporal_sim

# Training data: Image pack tfrecords list
tfdir = '/content/drive/MyDrive/cloud_mapping_planet_train/tfdata'
filenames1 = [os.path.join(tfdir, f) for f in os.listdir(tfdir) if f.endswith('1.tfrecords')]
list.sort(filenames1)
filenames2 = [os.path.join(tfdir, f) for f in os.listdir(tfdir) if f.endswith('2.tfrecords')]
list.sort(filenames2)
filenames3 = [os.path.join(tfdir, f) for f in os.listdir(tfdir) if f.endswith('3.tfrecords')]
list.sort(filenames3)

def train(epoch):
    filenames = [filenames1, filenames2, filenames3]

    # three packs will be zipped in the training pipeline
    train_ds = input_pipeline(filenames, batch_size, is_repeat=False)
    batchs = train_ds.batch(batch_size=batch_size)
    for epoch in range(epoch):
        loss_ = []
        loss_rec_ = []
        loss_sp_ = []
        loss_tp_ = []

        for train_pack1, train_pack2, train_pack3 in batchs:
            loss, loss_rec, loss_sp, loss_tp = train_step(train_pack1, train_pack2, train_pack3)
            loss_.append(loss)
            loss_rec_.append(loss_rec)
            loss_sp_.append(loss_sp)
            loss_tp_.append(loss_tp)

        print('train total loss over epoch %d: %.4f'% (epoch+1, np.mean(loss_)))
        print('train rec loss over epoch %d: %.4f'% (epoch+1, np.mean(loss_rec_)))
        print('train spatial loss over epoch %d: %.4f'% (epoch+1, np.mean(loss_sp_)))
        print('train temporal loss over epoch %d: %.4f'% (epoch+1, np.mean(loss_tp_)))

        if (epoch + 1) % 1 == 0:
            u_model.save_weights('u_model_autocm')
            d_model.save_weights('d_model_autocm')

train(100)
