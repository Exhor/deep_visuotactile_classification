import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image, ImageDraw
import re
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.losses import categorical_crossentropy
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def vt60_vision_data(width=100, encode=lambda x:x, encoded_dims=0, blotch=False):
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
                #print('.', end='')
                pic = Image.open(instpath + paths[img]).resize((width, width))
                if blotch:
                    draw = ImageDraw.Draw(pic)
                    a = 0.2 # pixels to cover
                    while True:
                        toblack = np.count_nonzero( np.sum(np.array(pic),axis=2) ) - (1-a) * pic.height * pic.width
                        print(toblack)
                        if toblack < 900:
                            break
                        radius = np.random.randint(10, 10+1.1*(toblack / 3.1416)**0.5)
                        cx = np.random.randint(radius+2, pic.width-radius-1)
                        cy = np.random.randint(radius+2, pic.height-radius-1)
                        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill='black', outline='black')

                imarray = np.expand_dims(np.array(pic), axis=0)
                v[label, instance, img] = encode(imarray)
    return v

def split_by_instance(v, test_instance, n_imgs_train, validation_instance=-1):
    n_labels, n_instances, n_imgs_test = v.shape[:3]
    assert(n_imgs_test >= n_imgs_train)
    fdim = v.shape[3:]
    reserved = 1
    if validation_instance > -1:
        reserved = 2
    xtr = np.zeros((n_labels * (n_instances - reserved) * n_imgs_train,) + fdim)
    ytr = np.zeros((n_labels * (n_instances - reserved) * n_imgs_train, n_labels))
    xte = np.zeros((n_labels * 1 * n_imgs_test,) + fdim)
    yte = np.zeros((n_labels * 1 * n_imgs_test, n_labels))
    xva = np.zeros((n_labels * 1 * n_imgs_test,) + fdim)
    yva = np.zeros((n_labels * 1 * n_imgs_test, n_labels))
    itr, ite, iva = 0, 0, 0
    for label in range(n_labels):
        for inst in range(n_instances):
            for imgid in range(n_imgs_test):
                if inst == test_instance:
                    xte[ite] = v[label, inst, imgid]
                    yte[ite] = to_categorical(label, n_labels)
                    ite += 1
                elif inst == validation_instance:
                    xva[iva] = v[label, inst, imgid]
                    yva[iva] = to_categorical(label, n_labels)
                    ite += 1
                else:
                    if imgid < n_imgs_train:
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

def m2_latefusion(vshape, tshape, optimiser='adadelta'):
    xin = Input((vshape[0]+tshape[0],))
    x = Dense(256, activation='relu')(xin)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.25)(x)
    xout = Dense(10, activation='softmax')(x)
    m = Model(xin, xout)
    m.compile(optimizer=optimiser, loss=categorical_crossentropy, metrics=['accuracy'])
    return m

def vt60_touch_data_aligned(v, ntouches=1):
    ''' Load VT60 database touch data and align it with visual data <v> '''
    n_labels = 10
    n_instances = 6
    n_train_t = 60
    n_v_imgs = v.shape[2]
    tdata = np.load('cache/vt60_touch_encoded_m4dcae.npy')
    d = tdata[()] # unpack dict
    n_obj, n_t_imgs = d['xtrain'].shape[:2]
    encoded_touch_dim = np.prod(d['xtrain'].shape[2:]) * ntouches
    vt = np.zeros((n_labels,
                  n_instances,
                  n_v_imgs,
                  25088 + encoded_touch_dim))
    for obj in range(n_obj):
        f = d['files_train'][obj]
        # '/home/tadeo/a2/code/data/vt60/touch/05_shoe/01/'
        s = [m.start() for m in re.finditer('/', f)]
        instance = int(f[s[-2]+1:s[-1]]) - 1
        label = int(f[s[-3]+1:s[-3]+3]) - 1
        tbunch = [d['xtrain'][obj][np.random.permutation(n_train_t)[:ntouches]].flatten() for i in range(n_v_imgs)]
        vt[label, instance] = np.concatenate((v[label, instance], np.array(tbunch)), axis=1)
    return vt


from keras.callbacks import LambdaCallback

if __name__ == '__main__':
    blotch = False
    fusion = True
    ntouches_allowed = [1, 3, 5, 10, 15, 20]

    # ktf.set_session(get_session()) # GPU vram use: grow as needed instead of hogging all at start
    # base = VGG16(weights='imagenet')
    # f_model = Model(inputs=base.input, outputs=base.get_layer('flatten').output)
    # f = lambda img : f_model.predict(preprocess_input(img.astype('float32')))
    # v = vt60_vision_data(width = 224, encode=f, encoded_dims=(25088,), blotch=blotch)
    # np.save('cache/vision_vgg16_encoded' + '_blotched'*blotch, v)
    v = np.load('cache/vision_vgg16_encoded' + '_blotched'*blotch + '.npy')
    optimiser = 'adadelta'
    epochs = 20
    n_testmigs = 40
    for ntouches in ntouches_allowed:
        vt = vt60_touch_data_aligned(v, ntouches=ntouches)
        for n_trainimgs in [40, 30, 20, 15, 10, 8, 5, 3, 2, 1]:
            print('Finetuning VGG16 using %s. Training with %d images.' % (optimiser, n_trainimgs))
            if fusion:
                print('ntouches = ', ntouches)
            n_test = 10 * n_testmigs # number of test samples
            cm = np.zeros((60, 10))  # confusion matrix objid -> pred class
            ypred = np.zeros((6, n_test), dtype='uint8')
            ytrue = np.zeros((6, n_test), dtype='uint8')
            vprob = np.zeros((6, n_test, 10)) # p(c|v)
            for test_instance in range(6):

                    # train
                    if fusion:
                        m = m2_latefusion(vshape=(25088,), tshape=(36*ntouches,), optimiser=optimiser)
                        x = vt
                    else:
                        x = v
                        m = m1_dense((25088,), optimiser)

                    xtr, ytr, xte, yte, xva, yva = split_by_instance(x, test_instance=test_instance,
                                                                     n_imgs_train=n_trainimgs)

                    cb = LambdaCallback(on_epoch_end= lambda epoch,logs: print('tr_acc = ', logs.get('acc'), ' # val_acc = ', logs.get('val_acc'))) # show progress
                    m.fit(xtr, ytr, batch_size=40, epochs=epochs, shuffle=True, verbose=0, validation_split=0.1, callbacks=[cb])

                    # test & record
                    vp = m.predict(xte)
                    vprob[test_instance] = vp
                    ypred[test_instance] = np.argmax(vp, axis=1)
                    ytrue[test_instance] = np.argmax(yte, axis=1)
                    print()
                    print('Test accuracy for instance %d: %.3f' % (test_instance, np.mean(ypred[test_instance]==ytrue[test_instance])))
                    for i in range(n_test):
                        cm[ytrue[test_instance,i]*6 + test_instance, ypred[test_instance,i]] += 1
            scipy.io.savemat('cache/vt_fusion_vgg16_m4dcae_256_32_%s_epochs_%d_ntrainv_%d_ntouch_%d_%s.mat' % (optimiser, epochs, n_trainimgs,ntouches, '_blotched'*blotch), {'ypred':ypred, 'ytrue':ytrue, 'cm':cm, 'vprob':vprob})
