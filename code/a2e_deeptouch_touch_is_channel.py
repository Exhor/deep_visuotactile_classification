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

# def m8_vtfusion(touchin_shape, visin_shape)
    # # touch layer
    #     x = Input(shape=input_shape)
    #     tin += [x]
    #     x = Conv2D(32, (3,3), activation='relu')(x)
    #     x = MaxPooling2D(pool_size=(2, 2))(x)
    #     x = Conv2D(32, (3,3), activation='relu')(x)
    #     x = MaxPooling2D(pool_size=(2, 2))(x)
    #     x = Dropout(0.25)(x)
    #     touchnets += [x]
    # x = keras.layers.concatenate(touchnets)
    # x = Flatten()(x)
    # x = Dense(50, activation='relu')(x)
    # x = Dense(50, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # out = Dense(output_shape, activation='softmax')(x)
    # model = Model(inputs=tin,outputs=[out])
    # model.compile(optimizer='adadelta', loss='binary_crossentropy')

def m7_dcae_plus_dense(input_shape, filters):
    ''' Various topologies (simplifications of this net) tested on 20touch set. This is best'''
    f = filters
    input_img = Input(shape=input_shape)
    x = Conv2D(f[0], (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(f[1], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(f[2], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(f[3], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(36, activation='relu')(x)
    x = Dropout(0.25)(x)
    # x = Dense(36, activation='relu')(x)
    # x = Dropout(0.25)(x)
    x = Dense(18, activation='relu')(x)
    xout = Dense(10, activation='softmax')(x)
    m = Model(input_img, xout)
    m.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def m6_dense(input_shape):
    xin = Input(shape=input_shape)
    x = Flatten()(xin)
    x = Dense(36, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(36, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(18, activation='relu')(x)
    xout = Dense(10, activation='softmax')(x)
    m = Model(xin, xout)
    m.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return m, xin, xout

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

def folder2tacvector(files_allowed, ntouches, centre, radius, width, enc=lambda x:x, enc_dims=0):
    newcentre = (width/2, width/2)
    newradius = width/2-2
    c = np.zeros((ntouches,) + enc_dims)
    #files = os.listdir(folder)
    rsample = np.random.permutation(len(files_allowed))
    for i in range(ntouches):
        path = files_allowed[rsample[i]]
        img = Image.open(path).convert('L')
        img = img.crop([centre[0]-radius,centre[1]-radius,
                        centre[0]+radius,centre[1]+radius])
        img = img.resize((width, width))
        imgarray = np.array(img)
        pol = cv2.linearPolar(imgarray, newcentre, newradius, flags=0)
        c[i] = enc(pol / 255)
    if np.any(np.isnan(c)):
        print('Error. NANs in polar image')
    return c

def vt60_touchdata(width = 96, test_on=[{1},{1},{1},{1},{1},{1},{1},{1},{1},{1}], n_train=60, n_test = 10, n_touches = 10, n_samples_per_object=1, enc=lambda x:x, enc_dims=0):
    if enc_dims==0:
        enc_dims = (width, width)
    num_classes = 10
    centre = (214,214)
    radius = 205


    vt60_touch = '/home/tadeo/a2/code/data/vt60/touch/'
    cpath = [p for p in os.listdir(vt60_touch) if os.path.isdir(vt60_touch+p)]
    folders_train = dict()
    folders_test = dict()
    for objClass in range(num_classes):
        train_on = {1, 2, 3, 4, 5, 6} - test_on[objClass]
        for instance in train_on:
            fpath = vt60_touch + cpath[objClass] + '/0' + str(instance) + '/'
            folders_train[objClass * 10 + instance] = fpath
        for instance in test_on[objClass]:
            fpath = vt60_touch + cpath[objClass] + '/0' + str(instance) + '/'
            folders_test[objClass * 10 + instance] = fpath

    x_train = np.zeros((n_samples_per_object*len(folders_train), n_touches,) + enc_dims)
    y_train = np.zeros((n_samples_per_object*len(folders_train), num_classes))
    f_train = ['' for i in range(x_train.shape[0])]
    x_test = np.zeros((n_samples_per_object*len(folders_test), n_touches,) + enc_dims)
    y_test = np.zeros((n_samples_per_object*len(folders_test), num_classes))
    f_test = ['' for i in range(x_test.shape[0])]
    i = 0
    for objID in folders_train:
        for n in range(n_samples_per_object):
            folder = folders_train[objID]
            files_allowed = [folder + f for f in os.listdir(folder)[:n_train]] # TODO: use only first 60? Randomise?
            x = folder2tacvector(files_allowed, n_touches, centre, radius, width, enc, enc_dims)
            x_train[i] = x
            y_train[i] = keras.utils.to_categorical(objID // 10, num_classes)
            f_train[i] = folder
            i += 1
    i = 0
    for objID in folders_test:
        for n in range(n_samples_per_object):
            folder = folders_test[objID]
            files_allowed = [folder + f for f in os.listdir(folder)[:n_test]]  # TODO: use only first 60? Randomise?
            x = folder2tacvector(files_allowed, n_touches, centre, radius, width, enc, enc_dims)
            x_test[i] = x
            y_test[i] = keras.utils.to_categorical(objID // 10, num_classes)
            f_test[i] = folder
            i += 1

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return x_train, y_train, x_test, y_test, f_train, f_test

def plot_recoded(encoder, x, width):

    n = 10  # how many digits we will display
    recoded_imgs = encoder.predict(x[:n])
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x[i].reshape(width, width))
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




from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    # pretraining
    filters = (32,16,8,4)
    width = 48
    cname = 'cache/m4_dcae_c%d(2)_c%d(2)_c%d(2)_c%d(2)_enc_ntr_60_w%d' % (filters + (width,))
    recalc = False
    #from keras.callbacks import TensorBoard
    if recalc:
        xtr, ytr, xte, yte, ftr, fte = vt60_touchdata(width=width, n_train=60, n_touches=1, n_samples_per_object=60)
        x1 = np.expand_dims(xtr[:, 0], -1)
        x2 = np.expand_dims(xte[:, 0], -1)
        encoder = m4_dcae(xtr.shape[2:] + (1,), filters)
        encoder.fit(x1, x1, epochs=100, batch_size=60, validation_data=(x2,x2))
        encoder.save(cname)
        plot_recoded(encoder, x2, width=width)
    else:
        encoder = keras.models.load_model(cname)

    # Train a model on the low level representations (encoded layer)
    f = Model(inputs=encoder.input, outputs=encoder.get_layer('encoded').output)
    encode = lambda img: f.predict(np.expand_dims(np.expand_dims(img, axis=0), axis=-1))
    acc = np.zeros((21, 21))

    # Just encode the database using m4_dcae
    ntouches = 120
    print('Encoding for ntouches = ' + str(ntouches))
    xtr, ytr, xte, yte, ftr, fte = vt60_touchdata(width=width, test_on=[set() for i in range(10)],
                                        n_train=120, n_touches=ntouches, n_samples_per_object=1,
                                        enc=encode, enc_dims=f.layers[-1].output_shape[1:])
    np.save('cache/vt60_touch_encoded_m4dcae', {'xtrain':xtr, 'ytrain':ytr, 'files_train':ftr})

    # for ntouches in [1,2,3,5,8,10,11,15,20]:
    # #ntouches = 3
    #     xtr, ytr, xte, yte = vt60_touchdata(width=width, n_train=60, n_touches=ntouches, n_samples_per_object=20, enc=encode, enc_dims=f.layers[-1].output_shape[1:])
    #
    #     cm = lambda a, b: confusion_matrix([np.argmax(h) for h in a], [np.argmax(h) for h in b])
    #
    #     for k in range(20):
    #         m = m6_dense(input_shape=xtr.shape[1:])
    #         m.fit(xtr, ytr, epochs=100, batch_size=50, validation_data=(xte, yte), verbose=0)
    #         q = cm(yte, m.predict(xte))
    #         print(k, '   #', np.trace(q) / sum(q.flatten()))
    #         acc[ntouches, k] = np.trace(q) / sum(q.flatten())
    #
    #     print(k, '#### mu=', np.mean(acc[ntouches]), '  sd=', np.std(acc[ntouches]))
    # np.save('cache/accs_c%d(2)_c%d(2)_c%d(2)_c%d(2)_enc_ntr_60_w%d_ntouches_2to15' % (filters + (width,)), acc)
