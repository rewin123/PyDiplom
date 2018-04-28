#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 23:00:25 2018

@author: yamaev
"""

import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import os

directory = 'Data'
if not os.path.exists(directory):
    os.makedirs(directory)

video_path = 'auchan2.avi'

cap = cv2.VideoCapture(video_path)
kernel_er = np.ones((3,3))
kernel = np.ones((11,11))
indexer = 0
while(cap.isOpened()):
    
    ret, frame = cap.read()
    dist = np.absolute(frame / 255 - fon)
    dist = dist[:,:,0] + dist[:,:,1] + dist[:,:,2]
    dist = dist * dist
    ret, dist = cv2.threshold(dist,0.4,1, cv2.THRESH_BINARY)
    dist = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel_er)
    dist = cv2.dilate(dist,kernel)
    label_image = label(dist)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 200:
            indexer += 1
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
#            cv2.imwrite(directory + '/' + str(indexer) + '.png',frame[minr:maxr, minc:maxc,:])
            cv2.rectangle(frame,(minc,minr),(maxc,maxr),(0,255,0),2)
            
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

