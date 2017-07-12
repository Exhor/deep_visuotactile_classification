# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import os
#
# gpu = False
# if not gpu:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras

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

def folder2tacvector(folder, ntouches, centre, radius, width, polarise=True):
    newcentre = (width/2, width/2)
    newradius = width/2-2
    c = np.zeros((ntouches, width, width))
    files = os.listdir(folder)
    rsample = np.random.permutation(min(ntouches, len(files)))
    for i in range(ntouches):
        path = folder + files[rsample[i]]
        img = Image.open(path).convert('L')
        img = img.crop([centre[0]-radius,centre[1]-radius,
                        centre[0]+radius,centre[1]+radius])
        img = img.resize((width, width))
        imgarray = np.array(img)
        if polarise:
            imgarray = cv2.linearPolar(imgarray, newcentre, newradius, flags=0)

        c[i,:,:] = imgarray
    if np.any(np.isnan(c)):
        print('Error. NANs in polar image')
    return c

def vt60_getparams(test_instances={1}):
    train_instances = {1, 2, 3, 4, 5, 6} - test_instances
    n_classes = 10
    centre = np.array([214,214])
    radius = 210

    vt60_touch = '/home/tadeo/a2/code/data/vt60/touch/'
    cpath = [p for p in os.listdir(vt60_touch) if os.path.isdir(vt60_touch+p)]
    folders_train = dict()
    folders_test = dict()
    for objClass in range(n_classes):
        for instance in train_instances:
            fpath = vt60_touch + cpath[objClass] + '/0' + str(instance) + '/'
            folders_train[objClass * 10 + instance] = fpath
        for instance in test_instances:
            fpath = vt60_touch + cpath[objClass] + '/0' + str(instance) + '/'
            folders_test[objClass * 10 + instance] = fpath
    return centre, radius, n_classes, folders_train, folders_test

def augment(filepath, nsamples, width, centre, radius, aug_shift):
    r = lambda h : np.random.randint(low=-aug_shift//2, high=aug_shift//2+1, size=2)
    c = centre + r(0) + r(0) # triangle distrib.
    x = folder2tacvector(filepath, nsamples, c, radius, width, polarise=True)
    return x

def touch_gen(x, y, ntouches, augs_per_instance=1):
    ''' Generates a pair x_train, y_train. Each pair has one sample per class per instance. A sample consists of
    ntouches feature vectors concatenated. A feature vector is x[aug,classinstance,n,...]'''
    if augs_per_instance > x.shape[0]:
        print('Warning: requested more augs than available')
        augs_per_instance = x.shape[0]
    while 1:
        augs = x.shape[0]
        nsamples = x.shape[2]
        rsample = np.random.permutation(nsamples)
        raugs = np.random.permutation(augs)[:augs_per_instance]
        xn = x[raugs][:,:,rsample[:ntouches]]
        xn = xn.reshape((augs_per_instance*x.shape[1],) + xn.shape[2:])
        yn = y[raugs,:,0]
        yn = yn.reshape((augs_per_instance*x.shape[1],) + yn.shape[2:])
        yield xn, yn

def vt60_touch_augment_encode(width=96, test_instances={1}, n_train=60, n_test=10, aug_shift=0, aug_factor=1,
                              enc=lambda h: h, enc_dim=0):
    if enc_dim==0:
        enc_dim = (width, width)

    centre, radius, num_classes, folders_train, folders_test = vt60_getparams(test_instances=test_instances)
    radius -= aug_shift

    # preprocess
    n_train_instances = len(folders_train)
    n_test_instances = len(folders_test)
    x_train = np.zeros((aug_factor, n_train_instances, n_train,) + enc_dim)
    y_train = np.zeros((aug_factor, n_train_instances, n_train, num_classes))
    x_test = np.zeros((1, n_test_instances, n_test,) + enc_dim)
    y_test = np.zeros((1, n_test_instances, n_test, num_classes))
    i = 0
    for tp in folders_train:
        folder = folders_train[tp]
        instance = tp % 10
        objclass = tp // 10
        for a in range(aug_factor):
            x_train[a][i] = enc(augment(folder, n_train, width, centre, radius, aug_shift))
            y_train[a][i] = np.tile(keras.utils.to_categorical(objclass, num_classes), (n_train,1))
        i += 1
    i = 0
    for tp in folders_test:
        folder = folders_test[tp]
        instance = tp % 10
        objclass = tp // 10
        x_test[0][i] = enc(augment(folder, n_test, width, centre, radius, 0))
        y_test[0][i] = np.tile(keras.utils.to_categorical(objclass, num_classes), (n_test,1))
        i += 1

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test

def vt60_touchdata(width=96, test_instances={1}, n_train=60, n_samples_train=1,
                   n_samples_test=1, n_test = 10, ntouches = 10,
                   enc=lambda x:x, aug_shift=0):
    centre, radius, num_classes, folders_train, folders_test = vt60_getparams(test_instances=test_instances)

    sample = enc(folder2tacvector(list(folders_train.values())[0], 1, centre, radius, width))
    enc_dims = sample.shape[1:]

    n = n_samples_train * len(folders_train)
    x_train = np.zeros((n, ntouches,) + enc_dims)
    y_train = np.zeros(n)

    n = n_samples_test * len(folders_test)
    x_test = np.zeros((n, ntouches,) + enc_dims)
    y_test = np.zeros(n)

    #x_train_gen = touch_generator(folders_train, n_train, ntouches, aug_shift, enc)

    i = 0
    for objID in folders_train:
        folder = folders_train[objID]
        x = folder2tacvector(folder, n_train, centre, radius, width, polarise=False)


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
    n = 10  # how many digits we will display
    recoded_imgs = encoder.predict(xte[:n])
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(xte[i].reshape(width, width))
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
    n_classes = 10
    n_instances = 6
    width = 48
    test_instances = {1}
    aug_shift = 10
    ntouches = 1 # pretraining
    xtr, ytr, xte, yte = vt60_touch_augment_encode(width=width, test_instances=test_instances, n_train=60, n_test=10,
                                                   aug_shift=aug_shift, aug_factor=1, enc=lambda x: x, enc_dim=0)
    # pretraining
    filters = (32,32,16,16)
    cname = 'cache/m4_dcae_c%d(2)_c%d(2)_c%d(2)_c%d(2)_enc_ntr_60_w%d' % (filters+(width,))
    recalc = True
    if recalc:
        encoder = m4_dcae(xtr.shape[-2:] + (1,), filters)
        xtr_flat = np.expand_dims(np.reshape(xtr, (np.prod(xtr.shape[:3]),) + xtr.shape[3:]), -1)
        xte_flat = np.expand_dims(np.reshape(xte, (np.prod(xte.shape[:3]),) + xte.shape[3:]), -1)
        #ytr_flat = np.reshape(ytr, (np.prod(xtr.shape[:3]), n_classes))
        #yte_flat = np.reshape(yte, (np.prod(xte.shape[:3]), n_classes))
        encoder.fit(xtr_flat, xtr_flat, batch_size=50, epochs=200, validation_data=(xte_flat, xte_flat))
        #encoder.fit(xtr[0], xtr[0], epochs=20, batch_size=60, validation_data=(xte[0],xte[0]))
        encoder.save(cname)
    else:
        encoder = keras.models.load_model(cname)
    plot_recoded(encoder, xte_flat, width=width)

    # Train a model on the low level representations (encoded layer)
    f = Model(inputs=encoder.input, outputs=encoder.get_layer('encoded').output)
    enc_dim = encoder.get_layer('encoded').output_shape[1:]
    enc = lambda imgs : f.predict(np.expand_dims(imgs,-1))

    ntouches = 10
    #xtr_enc, ytr, xte_enc, yte = vt60_touchdata(width=width, n_train=60, ntouches=ntouches,n_samples_train=1000, n_samples_test=20, enc=enc)
    xtr, ytr, xte, yte = vt60_touch_augment_encode(width=width, test_instances=test_instances, n_train=60, n_test=60,
                                                   aug_shift=aug_shift, aug_factor=100, enc=enc, enc_dim=enc_dim)
    xin = Input(shape=(ntouches,)+xtr.shape[3:])
    x = Flatten()(xin)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu')(x)
    xout = Dense(10, activation='softmax')(x)
    m = Model(xin, xout)
    m.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    trsample = np.random.permutation(xtr.shape[2])[:ntouches]
    tesample = np.random.permutation(xte.shape[2])[:ntouches]
    xtrc = xtr[:, :, trsample]
    xtec = xte[:, :, tesample]
    xtr_flat = xtrc.reshape((np.prod(xtrc.shape[:2]),) + xtrc.shape[2:])
    xte_flat = xtec.reshape((np.prod(xtec.shape[:2]),) + xtec.shape[2:])

    ytr_flat = np.reshape(ytr[:, :, 0], (ytr.shape[0] * ytr.shape[1], n_classes))
    yte_flat = np.reshape(yte[:, :, 0], (yte.shape[0] * yte.shape[1], n_classes))

    m.fit(xtr_flat, ytr_flat, batch_size=10, epochs=20, validation_data=(xte_flat, yte_flat))



