#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Create Train/Test splits (trials) for VT60 visual database.
Purpose is classification

@author: tadeo
"""
import random, os, numpy as np

def train_test_split(path, seed, ntrain=40, ntest=40):
    """ Returns (tr, te). Each tX[i = 0..ntrain/ntest][0=vis 1=tch][0=imgpath 1=imglabel] """
    random.seed(a=1) # repeatability
    test_instance = random.randint(1, 6)
    te = []
    tr = []
    class_paths = sorted(os.listdir(path + 'vision/')) # same for touch
    for c, cpath in enumerate(class_paths):
        for modality, mpath in enumerate(['vision/','touch/']):
            impaths = os.listdir(path + mpath + cpath + '/0' + str(test_instance))
            paths = np.random.choice(impaths, ntest, replace=False)
            te += [[impath, c+1] for impath in paths]
            for train_instance in range(1,7):
                if train_instance != test_instance:
                    impaths = os.listdir(path + mpath + cpath + '/0' + str(train_instance))
                    paths = np.random.choice(impaths, ntrain, replace=False)
                    tr += [[impath, c+1] for impath in paths]    
    return (tr, te)

if __name__ == '__main__':
    path = '/home/tadeo/p5/deep_visuotactile_classification/code/data/vt60/'
    seed = 1
    a = train_test_split(path, seed, ntrain=2, ntest=1)