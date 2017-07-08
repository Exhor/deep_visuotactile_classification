# -*- coding: utf-8 -*-
"""
Tests for capitaliser
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from scipy.interpolate import interp2d
from PIL import Image
import cv2

def polar2rect(img, c=[210,215], r=210):
    ''' disk inside img, centre c, radius r, transposed to rectangle [r, theta] '''
    img = img.crop(box=[c[0]-r,c[1]-r,c[0]+r+1,c[1]+r+1]).rotate(90)
    c = [r,r]
    i = np.array(img)
    rect = np.zeros((r, 360))
    f = interp2d(range(i.shape[0]), range(i.shape[1]), i.T)
    for rr in range(1,1+r):
        for theta in range(360):
            tx = c[0] + rr * np.cos(theta*np.pi/180)
            ty = c[1] + rr * np.sin(theta*np.pi/180)
            rect[rr-1,theta] = f(tx,ty)
    return rect, img

def test_p2c_matrix(img, c=[210,215], r=210):
    T = polar2cart_matrix(img, c=[210,215], r=210)
    v1 = polar2rect(img, c=[210,215], r=210)
    v2 = T * np.array(img)
    print(np.linalg.norm(T))

def polar2cart_matrix(img, c=[210,215], r=210):
    ''' Returns A, s.t. A*img = p, the polar image. c:centre, r:radius '''
    

if __name__ == '__main__':
    p = '/home/tadeo/a2/code/data/vt60/touch/01_stapler/01/touch_1_img_99_name_stapler_greyred_at_42094_5976_25275.jpg'
    #p = '/home/tadeo/a2/code/data/vt60/vision/03_ball/05/IMG_inst_5_img_10_validframeID_62.jpg'
    img = Image.open(p).convert('L')

    #r = tac2rect(img.crop([0,0,3,3]), c=[1,1], r=1)
    #r = tac2rect(img, c=[150,150], r=120)
    r, img = tac2rect(img)
    #fig,ax = plt.subplots(1)
    #ax.imshow(r)
    #dot = patches.Circle([210,215],radius=200)
    #ax.add_patch(dot)
    plt.imshow(img,cmap='gray')
    plt.show()    
    plt.imshow(r,cmap='gray')
    plt.show()
    #print(np.array(r))
