import os
import numpy as np
from PIL import Image
def vt60_vision_data():
    n_labels = 10
    n_instances = 6
    n_img = 40
    width = 100
    p = '/home/tadeo/a2/code/data/vt60/vision/'
    v = np.zeros((n_labels, n_instances, n_img, width, width, 3))
    for label in range(n_labels):
        s = ''
        if label < 10:
            s = '0'
        for instance in range(n_instances):
            path = os.listdir(p + s + str(label) + '/0' + str(instance) + '/.jpg')
            for img in range(n_img):
                img = Image.open(path).resize((width, width))
                v[label, instance, img] = np.array(img)
    return v

if __name__ == '__main__':
    a = vt60_vision_data()