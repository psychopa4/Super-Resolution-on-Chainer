# -*- coding: utf-8 -*-
import os
from os.path import join, exists
import glob
import six
import numpy as np
import math
import cv2
import random
from PIL import Image
import imageio
import scipy.misc
from chainer.dataset import dataset_mixin

def randomcrop(img, size=128):
    h,w,c=img.shape
    x = random.randint(0, h - size)
    y = random.randint(0, w - size)

    return img[x:x+size,y:y+size,:]

def augumentation(img):
    flipx=random.randint(0, 1)
    flipy=random.randint(0, 1)
    rot=random.randint(0, 1)
    if flipx==1:
        img=img[::-1,:,:]
    if flipy==1:
        img=img[::,::-1,:]
    if rot==1:
        img=np.transpose(img,(1,0,2))

    return img


class SingleImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, in_size=32, scale=4):
        self.paths=sorted(glob.glob(join(paths,'*.png')))
        self.in_size=in_size
        self.scale=scale
        self.out_size=in_size*scale

    def __len__(self):
        return len(self.paths)

    def get_example(self, i): #-> Image:
        path = self.paths[i]
        #HR = Image.open(path)
        HR = scipy.misc.imread(path)
        HR = randomcrop(HR, size=self.out_size)
        HR = augumentation(HR)

        h,w,c=HR.shape
        hr=Image.fromarray(HR)
        lr=hr.resize((h//self.scale,w//self.scale),Image.BICUBIC)

        LR=np.asarray(lr)
        LR=np.transpose(LR,(2,0,1))
        HR=np.transpose(HR,(2,0,1))
        return LR.astype(np.float32)/255., HR.astype(np.float32)/255.
        
