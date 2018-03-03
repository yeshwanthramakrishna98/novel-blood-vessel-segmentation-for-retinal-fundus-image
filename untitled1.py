# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:37:02 2018

@author: yeshwanth R
"""
from sys import exit
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import watershed, disk
from skimage import data
from skimage.io import imread
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import cv2
import os
import numpy as np
image = cv2.imread('25_training.tif')
kernel = np.ones((2,2), np.uint8)
kernel1 = np.ones((7,7), np.uint8)
kernel2 = np.ones((7,7), np.uint8)
g = image.copy()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0
img_grey = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
img_grey=(255-img_grey)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

cl1 = clahe.apply(img_grey)
cl1 = cv2.dilate(cl1,kernel,iterations = 1)

#cv2.imshow("op3",cl1)
erosion = cv2.erode(cl1,kernel1,iterations = 1)

erosion=cv2.GaussianBlur(erosion,(5,5),0)
erosion=cv2.blur(erosion,(5,5),0)
#erosion = cv2.medianBlur(cl1,1)
#opening = cv2.morphologyEx(cl1, cv2.MORPH_OPEN, kernel)
#cv2.imshow("op2",erosion)
#cv2.normalize(opening, opening, 0, 255, cv2.NORM_MINMAX)
#opening=(255-opening)
x=cv2.subtract(cl1,erosion)
x = cv2.medianBlur(x,5)
cv2.imshow("x",x)
x1 = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel2)
#cv2.imshow("op1",x1)
x2=cv2.subtract(x,x1)
cv2.imshow("x2",x2)
x3 = cv2.morphologyEx(x2, cv2.MORPH_OPEN, kernel2)
x4=cv2.subtract(x2,x3)
x5=cv2.add(x2,x4)
cv2.imshow("x5",x5)

image1 = img_as_ubyte(x5)
markers = rank.gradient(image.disk(5)) < 20
markers = ndi.label(markers)[0]

gradient = rank.gradient(image, disk(2))

labels = watershed(gradient, markers)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8), sharex=True,
                         sharey=True, subplot_kw={'adjustable':'box-forced'})

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax[2].set_title("Markers")

ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[3].imshow(labels, cmap=plt.cm.spectral, interplolation='nearest',alpha=.7)

ax[3].set_title("segmented")

for a in ax:
     a.axis('off')
     
     fig.tight_layout()
     plt.show()








#edges = cv2.Canny(x5,10,100)
#cv2.imshow("edges",edges)
#abc=cv2.subtract(x5,edges)
#cv2.imshow("abc",abc)
#cv2.waitKey(0)



