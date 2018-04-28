#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:20:43 2018

@author: yamaev
"""

import skimage as sk
import numpy as np
import scipy as sp
from skimage import data, io, filters, segmentation, color, draw
from skimage.future import graph
import cv2

def Projective(img, lbl):
    out = color.label2rgb(lbl, img, kind='avg')
    return out

n = 5

min_v = mean.min()
max_v = mean.max()

dv = max_v - min_v

segment = (mean - min_v) / dv * n
segment = segment.astype(int)
segment = segment[:,:,0] + segment[:,:,1] * n + segment[:,:,2] * n * n

fon = Projective(mean, segment)

cv2.imshow('frame', fon)