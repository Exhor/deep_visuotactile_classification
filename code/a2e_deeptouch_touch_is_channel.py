# -*- coding: utf-8 -*-
"""
Tests for capitaliser
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import os 

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Permute, UpSampling2D


#def m5_dae(input_shape):


def m4_dcae(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model

def m2(input_shape, output_shape):
    model = Sequential()
    model.add(SeparableConv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(SeparableConv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model

def m3_deepfusion(input_shape, output_shape, ntouches):
    tin = []
    touchnets = []
    for t in range(ntouches):
        x = Input(shape=input_shape)
        tin += [x]
        x = Conv2D(32, (3,3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3,3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        touchnets += [x]
    x = keras.layers.concatenate(touchnets)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(output_shape, activation='softmax')(x)
    model = Model(inputs=tin,outputs=[out])
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model


def m1(input_shape, output_shape):
    model = Sequential()

    model.add(SeparableConv2D(32, (3, 3), padding='valid',
                     #strides=(3, 3), # y-stride jumps between touch images
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(SeparableConv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    return model

def folder2tacvector(folder, ntouches, centre, radius, width):
    newcentre = (width/2, width/2)
    newradius = width/2-2
    c = np.zeros((width, width, ntouches))
    files = os.listdir(folder)
    for i in range(ntouches):
        path = folder + files[np.random.randint(len(files))]
        img = Image.open(path).convert('L')
        img = img.crop([centre[0]-radius,centre[1]-radius,
                        centre[0]+radius,centre[1]+radius])
        img = img.resize((width, width))
        imgarray = np.array(img)
        pol = cv2.linearPolar(imgarray, newcentre, newradius, flags=0)
        c[:,:,i] = pol
    if np.any(np.isnan(c)):
        print('Error. NANs in polar image')
    return c

def runit(test_on={1}, n_train=6, n_test = 10, epochs = 100):
    num_classes = 10
    ntouches = 10
    centre = (214,214)
    radius = 205
    width = 96
    train_on = {1,2,3,4,5,6} - test_on
    vt60_touch = '/home/tadeo/a2/code/data/vt60/touch/'
    cpath = [p for p in os.listdir(vt60_touch) if os.path.isdir(vt60_touch+p)]
    folders_train = dict()
    folders_test = dict()
    for objClass in range(num_classes):
        for instance in train_on:
            fpath = vt60_touch + cpath[objClass] + '/0' + str(instance) + '/'
            folders_train[objClass * 10 + instance] = fpath
        for instance in test_on:
            fpath = vt60_touch + cpath[objClass] + '/0' + str(instance) + '/'
            folders_test[objClass * 10 + instance] = fpath

    x_train = np.zeros((n_train*len(folders_train), width, width, ntouches))
    y_train = np.zeros(n_train*len(folders_train))

    x_test = np.zeros((n_test*len(folders_test), width, width, ntouches))
    y_test = np.zeros(n_test*len(folders_test))
    i = 0
    for objID in folders_train:
        for n in range(n_train):
            folder = folders_train[objID]
            x = folder2tacvector(folder, ntouches, centre, radius, width)
            x_train[i,:,:,:] = x
            y_train[i] = objID // 10
            i += 1
    i = 0
    for objID in folders_test:
        for n in range(n_test):
            folder = folders_test[objID]
            x = folder2tacvector(folder, ntouches, centre, radius, width)
            x_test[i,:,:,:] = x
            y_test[i] = objID // 10
            i += 1

    batch_size = 30
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # reshape input to go into multi-net
    x_train = [np.expand_dims(x_train[:, :, :, i],axis=-1) for i in range(ntouches)]
    x_test = [np.expand_dims(x_test[:, :, :, i],axis=-1) for i in range(ntouches)]

    # pretraining
    from keras.callbacks import TensorBoard
    model = m4_dcae(x_train[0].shape[1:])
    model.fit(x_train[0], x_train[0], epochs=50, batch_size=60, shuffle=True,
           validation_data=(x_test[0],x_test[0]),
           callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # model = m3_deepfusion(input_shape = x_train[0].shape[1:],
    #                       output_shape=num_classes,
    #                       ntouches=ntouches)
    


    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(x_test, y_test)
    #           )
    return model, x_train, y_train, x_test, y_test

if __name__ == '__main__':
    #ae = m4_dcae((96,96,1))
    #ae.summary()
    model, xtr, ytr, xte, yte = runit()
    print('a')