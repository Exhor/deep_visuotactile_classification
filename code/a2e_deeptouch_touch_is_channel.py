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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D

def m2(input_shape, output_shape) -> keras.models.Sequential:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    return model

def m1(input_shape, output_shape) -> keras.models.Sequential:
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='valid',
                     #strides=(3, 3), # y-stride jumps between touch images
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))
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

def run(test_on={1}, n_train=60, n_test = 10, epochs = 10):
    num_classes = 10
    ntouches = 1
    centre = (214,214)
    radius = 205
    width = 100
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

    batch_size = 60
    
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
    
    model = m2(input_shape = x_train.shape[1:],
               output_shape=num_classes)

    # initiate optimizer
    opt = keras.optimizers.Adadelta()
    #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test)
              )
    return model, x_test, y_test

if __name__ == '__main__':
    model = run()