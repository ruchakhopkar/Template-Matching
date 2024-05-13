#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:58:21 2023

@author: ruchak
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv

#load the template you wanto find
template = cv.imread('/home/ruchak/Desktop/image-classification/fesem/OneDrive_2023-03-28/FA TEP FESEM ML/AT4BI9IAC2-F-001.tiff',
                      cv.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]
# plt.imshow(template, cmap  = 'gray')
# plt.show()

methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

src = '/home/ruchak/Desktop/image-classification/fesem/OneDrive_2023-03-28/FA TEP FESEM ML/'
dest = '/home/ruchak/Desktop/image-classification/fesem/cropped/'
l1 = sorted(os.listdir('/home/ruchak/Desktop/image-classification/fesem/OneDrive_2023-03-28/FA TEP FESEM ML/'))
l1 = [x for x in l1 if '7' in x and '.tif' not in x]

#for all files in a certain directory, crop the templates from each image and save it
for directory in l1:
    for path, directories, files in os.walk(os.path.join(src, directory)):
          print(directory)
          for file in files:
              img2 = cv.imread(path + '/'+file,
                                cv.IMREAD_GRAYSCALE)
              h1, w1 = img2.shape
              if h1<=h:
                  img2 = cv.resize(img2, (w1, h+1), interpolation = cv.INTER_AREA)
              h1, w1 = img2.shape
              if w1<=w:
                  img2 = cv.resize(img2, (w+1, h1), interpolation = cv.INTER_AREA)
              for meth in methods:
                  img = img2.copy()
                  method = eval(meth)
                  # Apply template Matching
                  res = cv.matchTemplate(img,template,method)
                  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                  # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                  if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                      top_left = min_loc
                  else:
                      top_left = max_loc
                  bottom_right = (top_left[0] + w, top_left[1] + h)
                  # cv.rectangle(img,top_left, bottom_right, 255, 5)
                  # plt.subplot(121),plt.imshow(res,cmap = 'gray')
                  # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                  # plt.subplot(122),plt.imshow(img,cmap = 'gray')
                  # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                  # plt.suptitle(meth)
                  # plt.show()
                  # plt.imshow(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], cmap = 'gray')
                  # plt.show()
                 
                  cv.imwrite(dest+meth.split('.')[1]+ '/'+ directory+'/'+file[:-4]+'.png', img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
         
        
     
