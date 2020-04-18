#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:24:45 2020

Test script and visualistion of objectSegmentation.

@author: Jens Settelmeier
"""

from objectSegmentation import objectSegmentation
from matplotlib import pyplot as plt
import cv2
import numpy as np

# Path to the image
imgPath = './testdata/object1.jpg'

# get the segmentation of the reflector and the coordinates in the picture
coordinates,segmentation = objectSegmentation(imgPath)

# Plot the segmentation
plt.figure()
plt.imshow(segmentation)

# Plot the original image with the marker on the reflector.
img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.scatter(coordinates[0],coordinates[1],marker='+',color='r')
plt.imshow(img_RGB)
