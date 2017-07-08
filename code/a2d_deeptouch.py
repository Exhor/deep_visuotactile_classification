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
from keras.layers import Conv2D, MaxPooling2D


def folder2tacvector(folder, ntouches, centre, radius, width):
    newcentre = (width/2, width/2)
    newradius = width/2-2
    c = np.zeros((ntouches*width, width))
    files = os.listdir(folder)
    for i in range(ntouches):
        path = folder + files[np.random.randint(len(files))]
        img = Image.open(path).convert('L')
        img = img.crop([centre[0]-radius,centre[1]-radius,
                        centre[0]+radius,centre[1]+radius])
        img = img.resize((width, width))
        pol = cv2.linearPolar(np.array(img), newcentre, newradius, flags=0)
        #pol = (pol - pol.mean(axis=0)) / pol.std(axis=0)
        c[i*width:(i+1)*width,:] = pol.T
    if np.any(np.isnan(c)):
        print('Error. NANs in polar image')
    return c, pol

if __name__ == '__main__':
    ntouches = 3
    centre = (214,214)
    radius = 205
    width = 100
    
    folders_train = {1:'/home/tadeo/a2/code/data/vt60/touch/01_stapler/01/',
                     11:'/home/tadeo/a2/code/data/vt60/touch/02_bottleempty/01/'}
    folders_test = {2:'/home/tadeo/a2/code/data/vt60/touch/01_stapler/02/',
                    12:'/home/tadeo/a2/code/data/vt60/touch/02_bottleempty/02/'}
    n_train = 100
    x_train = np.zeros((n_train*len(folders_train), width, width, ntouches))
    y_train = np.zeros(n_train*len(folders_train))
    n_test = 1
    x_test = np.zeros((n_test*len(folders_test), width, width, ntouches))
    y_test = np.zeros(n_test*len(folders_test))
    i = 0
    for objID in folders_train:
        for n in range(n_train):
            folder = folders_train[objID]
            x = folder2tacvector(folder, ntouches, centre, radius, width)
            x_train[i,:,:] = x
            y_train[i] = objID // 10
            i += 1
    i = 0
    for objID in folders_test:
        for n in range(n_test):
            folder = folders_test[objID]
            x = folder2tacvector(folder, ntouches, centre, radius, width)
            x_test[i] = x
            y_test[i] = objID // 10
            i += 1
    
    
    batch_size = 32
    num_classes = 10
    epochs = 10
    data_augmentation = False
    stride_x = 1 # cnn rotational stride, max = width
    
    # The data, shuffled and split between train and test sets:
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    xtr = x_train
    xte = x_test
    s = x_train.shape
    x_train = np.zeros((s[0],s[1],s[2],3))
    s = x_test.shape
    x_test = np.zeros((s[0],s[1],s[2],3))
    for i in range(3):
        x_train[:,:,:,i] = xtr
        x_test[:,:,:,i] = xte
    
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, width), padding='same',
                     strides=(stride_x, width), # y-stride jumps between touch images
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
    
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
    validation_data=(x_test, y_test))
        