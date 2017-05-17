#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Create Train/Test splits (trials) for VT60 visual database.
Purpose is classification

@author: tadeo
"""
import random
if __name__ == '__main__':
    random.seed(a=1) # repeatability
    fname = 'train-test_splits.txt'
    t = dict()  # t[ntrial][0][i][0] is the ith file for training
                # t[ntrial][1][i][0] is the ith file for testing
                # ...............[1] is the label
    ntrials = 100
    for trial in range(ntrials):
        test_instance = random.randint(1,6)
        
        