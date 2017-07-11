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
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Permute, UpSampling2D
import theano

#def m5_dae(input_shape):

def m5_dae(input_shape):
    input_img = Input(shape=input_shape)
    x = Flatten()(input_img)
    x = Dense(512, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(16, activation='relu', name='encoded')(x)

    x = Dense(64, activation='relu')(encoded)
    x = Dense(512, activation='relu')(x)
    x = Dense(np.prod(input_shape), activation='sigmoid')(x)
    decoded = Reshape(input_shape)(x)
    model = Model(input_img, decoded)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model

def m4_dcae(input_shape, filters=(16,8,4,1)):
    f = filters
    input_img = Input(shape=input_shape)
    x = Conv2D(f[0], (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(f[1], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(f[2], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(f[3], (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same', name='encoded')(x)

    #encoded = Dense(20, activation='sigmoid')(x)

    x = Conv2D(f[3], (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(f[2], (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(f[1], (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(f[0], (3, 3), activation='relu', padding='same')(x)
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
    c = np.zeros((ntouches, width, width))
    files = os.listdir(folder)
    for i in range(ntouches):
        path = folder + files[np.random.randint(len(files))]
        img = Image.open(path).convert('L')
        img = img.crop([centre[0]-radius,centre[1]-radius,
                        centre[0]+radius,centre[1]+radius])
        img = img.resize((width, width))
        imgarray = np.array(img)
        pol = cv2.linearPolar(imgarray, newcentre, newradius, flags=0)
        c[i,:,:] = pol
    if np.any(np.isnan(c)):
        print('Error. NANs in polar image')
    return c

def vt60_touchdata(width=96, test_on={1}, n_train=60, n_samples_train=1,
                   n_samples_test=1, n_test = 10, ntouches = 10, enc=lambda x:x):
    num_classes = 10
    centre = (214,214)
    radius = 205
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

    sample = enc(folder2tacvector(fpath, 1, centre, radius, width))
    enc_dims = sample.shape[1:]

    n = n_samples_train * len(folders_train)
    x_train = np.zeros((n, ntouches,) + enc_dims)
    y_train = np.zeros(n)

    n = n_samples_test * len(folders_test)
    x_test = np.zeros((n, ntouches,) + enc_dims)
    y_test = np.zeros(n)
    i = 0
    for objID in folders_train:
        folder = folders_train[objID]
        x = enc(folder2tacvector(folder, n_train, centre, radius, width))
        for n in range(n_samples_train):
            x_sample = x[np.random.permutation(n_train)[:ntouches],:,:]
            x_train[i] = x_sample
            y_train[i] = objID // 10
            i += 1
    i = 0
    for objID in folders_test:
        folder = folders_test[objID]
        x = enc(folder2tacvector(folder, n_test, centre, radius, width))
        for n in range(n_samples_test):
            x_sample = x[np.random.permutation(n_test)[:ntouches],:,:]
            x_test[i] = x_sample
            y_test[i] = objID // 10
            i += 1
    
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
    x_train = np.expand_dims(x_train,axis=-1)
    x_test = np.expand_dims(x_test,axis=-1)
    return x_train, y_train, x_test, y_test

def plot_recoded(encoder, xte, width):
    recoded_imgs = encoder.predict(xte[:,0,:,:])
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(xte[i, 0, :, :].reshape(width, width))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recoded_imgs[i].reshape(width, width))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def save_model(m, path):
    m_json = m.to_json()
    with open(path+'.json', 'w') as jfile:
        jfile.write(m_json)
    m.save_weights(path+'h5')

from keras.models import model_from_json

def load_model(path):
    jfile = open(path+'.json','r')
    mjson = jfile.read()
    jfile.close()
    m = model_from_json(mjson)
    m.load_weights(path+'h5')
    return m

if __name__ == '__main__':
    width = 48
    xtr, ytr, xte, yte = vt60_touchdata(width=width, n_train=60, ntouches=1)

    # pretraining
    filters = (20,15,6,3)
    cname = 'cache/m4_dcae_c%d(2)_c%d(2)_c%d(2)_c%d(2)_enc_ntr_60_w48' % filters
    recalc = False
    #from keras.callbacks import TensorBoard
    if recalc:
        encoder = m4_dcae(xtr[0].shape[1:], filters)
        encoder.fit(xtr[0], xtr[0], epochs=20, batch_size=60, validation_data=(xte[0],xte[0]))
        encoder.save(cname)
    else:
        encoder = keras.models.load_model(cname)
    plot_recoded(encoder, xte, width=width)

    # Train a model on the low level representations (encoded layer)
    f = Model(inputs=encoder.input, outputs=encoder.get_layer('encoded').output)
    enc = lambda imgs : f.predict(np.expand_dims(imgs,-1))

    ntouches = 5
    xtr_enc, ytr, xte_enc, yte = vt60_touchdata(width=width, n_train=60, ntouches=ntouches,
                                                n_samples_train=1000, n_samples_test=20, enc=enc)

    xin = Input(shape=xtr_enc.shape[1:])
    x = Flatten()(xin)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu')(x)
    xout = Dense(10, activation='softmax')(x)
    m = Model(xin, xout)
    m.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    m.fit(xtr_enc, ytr, epochs=20, batch_size=60, validation_data=(xte_enc,yte))



    # model = m3_deepfusion(input_shape = x_train[0].shape[1:],
    #                       output_shape=num_classes,
    #                       ntouches=ntouches)
