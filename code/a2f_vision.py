import os
import numpy as np
import scipy.io
from PIL import Image

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.losses import categorical_crossentropy
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def vt60_vision_data(width=100, encode=lambda x:x, encoded_dims=0):
    n_labels = 10
    n_instances = 6
    n_imgs = 40
    if encoded_dims==0:
        encoded_dims = (width, width)

    p = '/home/tadeo/a2/code/data/vt60/vision/'
    cfolders = os.listdir(p)

    v = np.zeros((n_labels,
                  n_instances,
                  n_imgs,) +
                 encoded_dims)

    for label in range(n_labels):
        print('')
        print('Preprocessing class %d of %d' % (label, 10))
        cfolder = min(x for x in cfolders if str(label+1) in x)
        for instance in range(n_instances):
            instpath = p + cfolder + '/0' + str(instance+1) + '/'
            paths = [x for x in os.listdir(instpath) if x[-4:] == '.jpg']
            for img in range(n_imgs):
                print('.', end='')
                pic = Image.open(instpath + paths[img]).resize((width, width))
                imarray = np.expand_dims(np.array(pic), axis=0)
                v[label, instance, img] = encode(imarray)
    return v

def split_by_instance(v, test_instance, validation_instance=-1):
    n_labels, n_instances, n_imgs = v.shape[:3]
    fdim = v.shape[3:]
    reserved = 1
    xtr = np.zeros((n_labels * (n_instances - reserved) * n_imgs,) + fdim)
    ytr = np.zeros((n_labels * (n_instances - reserved) * n_imgs, n_labels))
    xte = np.zeros((n_labels * 1 * n_imgs,) + fdim)
    yte = np.zeros((n_labels * 1 * n_imgs, n_labels))
    xva = np.zeros((n_labels * 1 * n_imgs,) + fdim)
    yva = np.zeros((n_labels * 1 * n_imgs, n_labels))
    itr, ite, iva = 0, 0, 0
    for label in range(n_labels):
        for inst in range(n_instances):
            for imgid in range(n_imgs):
                if inst == test_instance:
                    xte[ite] = v[label, inst, imgid]
                    yte[ite] = to_categorical(label, n_labels)
                    ite += 1
                elif inst == validation_instance:
                    xva[iva] = v[label, inst, imgid]
                    yva[iva] = to_categorical(label, n_labels)
                    ite += 1
                else:
                    xtr[itr] = v[label, inst, imgid]
                    ytr[itr] = to_categorical(label, n_labels)
                    itr += 1
    # Shuffle data
    tri = np.random.permutation(xtr.shape[0])
    tei = np.random.permutation(xte.shape[0])
    vai = np.random.permutation(xva.shape[0])
    return xtr[tri], ytr[tri], xte[tei], yte[tei], xva[vai], yva[vai]

def m1_dense(inshape, optimiser='adadelta'):
    m = Sequential()
    m.add(Dense(256, activation='relu', input_shape=inshape))
    m.add(Dropout(0.25))
    m.add(Dense(32, activation='relu'))
    m.add(Dropout(0.25))
    # m.add(Dense(32, activation='relu'))
    # m.add(Dropout(0.25))
    m.add(Dense(10, activation='softmax'))
    m.compile(optimizer=optimiser, loss=categorical_crossentropy, metrics=['accuracy'])
    return m

if __name__ == '__main__':
    # ktf.set_session(get_session()) # GPU vram use: grow as needed instead of hogging all at start
    # base = VGG16(weights='imagenet')
    # f_model = Model(inputs=base.input, outputs=base.get_layer('flatten').output)
    # f = lambda img : f_model.predict(preprocess_input(img.astype('float32')))
    # v = vt60_vision_data(width = 224, encode=f, encoded_dims=(25088,))
    # np.save('cache/vision_vgg16_encoded.np', v)
    v = np.load('cache/vision_vgg16_encoded.np.npy')
    cm = np.zeros((60,10)) # confusion matrix objid -> pred class
    optimiser = 'adadelta'
    epochs = 50
    print('Finetuning using %s' % optimiser)
    n = 400 # number of test samples TODO: automate
    ypred = np.zeros((6, n), dtype='uint8')
    ytrue = np.zeros((6, n), dtype='uint8')
    vprob = np.zeros((6, n, 10)) # p(c|v)
    for test_instance in range(6):
        xtr, ytr, xte, yte = split_by_instance(v, test_instance=test_instance)
        # train
        m = m1_dense(xtr.shape[1:], optimiser)

        m.fit(xtr, ytr, batch_size=40, epochs=epochs, shuffle=True, verbose=1, validation_split=0.1)
        # test & record
        vp = m.predict(xte)
        vprob[test_instance] = vp
        ypred[test_instance] = np.argmax(vp, axis=1)
        ytrue[test_instance] = np.argmax(yte, axis=1)
        print('Test accuracy for instance %d: %.3f' % (test_instance, np.mean(ypred[test_instance]==ytrue[test_instance])))
        for i in range(n):
            cm[ytrue[test_instance,i]*6 + test_instance, ypred[test_instance,i]] += 1
        #np.save('cache/vision_vgg16_pred_true_instance_%d' % test_instance, (ypred, ytrue))
    scipy.io.savemat('cache/vision_vgg16_256_32_%s_epochs_%d_pred_true_cm_pred_vprob.mat' % (optimiser, epochs),
                     {'ypred':ypred, 'ytrue':ytrue, 'cm':cm, 'vprob':vprob})
    #np.save('cache/vision_vgg16_cm', cm)

