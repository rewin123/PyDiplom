#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:11:55 2018

@author: yamaev
"""

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

def create_dense_ae():
    input_img = Input(shape=(28, 28, 3))

    x = Conv2D(256, (7, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (7, 7), activation='linear', padding='same')(x)

    # На этом моменте представление  (7, 7, 1) т.е. 49-размерное

    input_encoded = Input(shape=(7, 7, 1))
    x = Conv2D(64, (7, 7), activation='linear', padding='same')(input_encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (7, 7), activation='sigmoid', padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

def Images(path, width, height):
    all_images = []
    for image_path in os.listdir(path):
        img = cv2.imread(path + '/' + image_path)
        if not img is None :
            if(img.shape[0] > 0):
                img = cv2.resize(img, (width,height))
                img = img / 255
                all_images.append(img)
    return np.array(all_images)

def Test(autoencoder,ds, pos):
    result = autoencoder.predict(ds[pos:pos + 1,:,:,:])
    cv2.imshow('frame',ds[pos])
    cv2.imshow('frame2', result[0])

batch_size = 16



# this is a similar generator, for validation data
#validation_generator = test_datagen.flow_from_directory(
#        'data/validation',
#        target_size=(150, 150),
#        batch_size=batch_size,
#        class_mode='binary')
##
encoder, decoder, autoencoder = create_dense_ae()
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
