# -*- coding: utf-8 -*-
"""
Reshape old style tactile database (filename structure) to (folder structure)
"""
import os, re, shutil
if __name__ == '__main__':
    loc = '/home/tadeo/data/c10/c10/c10_dry_touch/'
    d = os.listdir(loc)
    for f in d:
        if not os.path.isdir(loc+f) and '_' in f:
            ci = int(f[f.find('_')+1:f.find('_',7)])
            cla = ci // 100 + 1
            class_name = str(cla) + '_'
            if len(class_name) < 3:
                class_name = '0' + class_name
            class_name += f[f.find('_name_')+6:f.find('_',f.find('_name_')+6)]
            ins = ci % 100
            img = int(f[f.find('_img_')+5:f.find('_name_')])
            #print loc + f + '->' + loc + class_name + '/0' + str(ins) + '/' + f
            #os.mkdir(loc + class_name)
            fdir = loc + class_name + '/0' + str(ins)
            if not os.path.isdir(fdir):
                os.makedirs(fdir)
            shutil.copyfile(loc + f, fdir + '/' + f)
            