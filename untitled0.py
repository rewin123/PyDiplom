#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:45:53 2018

@author: yamaev
"""
from skimage import data, io, filters, segmentation, color, draw
import numpy as np
import cv2


def Show(img):
    io.imshow(img)

video_path = 'auchan2.avi'

cap = cv2.VideoCapture(video_path)

width = int(cap.get(3))
height = int(cap.get(4))
#mean = np.zeros([height,width])
position = 0
while(cap.isOpened()):
    position += 1
    print(position)
    ret, frame = cap.read()
    if position == 1:
        mean = np.array(frame.shape)
    
    if ret:
        mean = mean + frame
    else:
        break

#    cv2.imshow('frame',mean / mean.max())
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break


mean = mean / position / 255
cv2.imshow('frame', mean)

