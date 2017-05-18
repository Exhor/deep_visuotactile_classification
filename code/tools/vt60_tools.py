#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Create Train/Test splits (trials) for VT60 visual database.
Purpose is classification

@author: tadeo
"""
import random, os, numpy as np



def train_test_split(path, seed, test_instance, touches_per_photo=5, n_photos_train=40, n_photos_test=1):
    """ Returns (tr, te). Each tX = [[label, photopath, touchpaths] for i in nclasses*n_photos_train/test] """
    random.seed(a=1) # repeatability
    te = []
    tr = []
    class_paths = sorted(os.listdir(path + 'vision/')) # same for touch
    for c, cpath in enumerate(class_paths):
        #for modality, mpath in enumerate(['vision/','touch/']):
        vpath = path + 'vision/' + cpath + '/0' + str(test_instance) + '/'
        tpath = path + 'touch/' + cpath + '/0' + str(test_instance) + '/'
        photo_paths_all = os.listdir(vpath)
        touch_paths_all = os.listdir(tpath)
        photo_paths = np.random.choice(photo_paths_all, n_photos_test, replace=False)
        for photo_path in photo_paths:
            touch_paths = np.random.choice(touch_paths_all, touches_per_photo, replace=False)
            te += [[c, vpath+photo_path, [tpath+p for p in touch_paths]]]
            
        for train_instance in range(1,7):
            vpath = path + 'vision/' + cpath + '/0' + str(train_instance) + '/'
            tpath = path + 'touch/' + cpath + '/0' + str(train_instance) + '/'
            photo_paths_all = os.listdir(vpath)
            touch_paths_all = os.listdir(tpath)
            photo_paths = np.random.choice(photo_paths_all, n_photos_train, replace=False)
            for photo_path in photo_paths:
                touch_paths = np.random.choice(touch_paths_all, touches_per_photo, replace=False)
                tr += [[c, vpath+photo_path, [tpath+p for p in touch_paths]]]
    return (tr, te)

if __name__ == '__main__':
    path = '/home/tadeo/p5/deep_visuotactile_classification/code/data/vt60/'
    seed = 1
    a = train_test_split(path, seed, test_instance=1, touches_per_photo=2, n_photos_train=2, n_photos_test=1)