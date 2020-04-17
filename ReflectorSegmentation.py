#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:03:31 2020

@author: Jens Settelmeier

Function that can find the coordinates of an item, that is surrounded by the 
color red, as long as there is not to much red in any other regions of the 
picture and the light is not too bright.
Further the object is also segmentated.

Input: imgPath: Path to Image
Ouput: reflector_coordinates: Coordinates of red surrounded object 
       reflector_segmentation: Image with the segmentated region of the object
    
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

def reflectorSegmentation(imgPath):

    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    
    img_filttered = img
    
    # noise reduction
    for i in range(3):
        img_filttered[:,:,i] = cv2.GaussianBlur(img[:,:,i],(5,5),0)
        
    hsv = cv2.cvtColor(img_filttered, cv2.COLOR_BGR2HSV)
       
    hsv_filttered = hsv
    for i in range(3):
        hsv_filttered[:,:,i] = cv2.GaussianBlur(hsv[:,:,i],(5,5),0)
        
    hChannel = hsv_filttered[:,:,0]
    out = np.zeros(hChannel.shape, np.uint8)
    normalized = cv2.normalize(hChannel, out, 0, 255, cv2.NORM_MINMAX)
    tolerance = 15;
    
    ref_reflector = np.max(normalized)  # has to be 255
    
    normalized[np.nonzero(normalized>=ref_reflector - tolerance)]=255
    normalized[np.nonzero(normalized<ref_reflector - tolerance)]=0
    
    im_floodfill = normalized.copy()
    # construct mask
    h, w = normalized.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
       
    # fill "holes"
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    reflector_segmentation = cv2.bitwise_not(im_floodfill)
    
    max_val = np.max(reflector_segmentation) # still 255 if not debug
     
    # find centers with K-Means
    [row, col] = np.nonzero(reflector_segmentation==max_val)   
    X = np.transpose(np.array([row,col]))
    kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
    
    reflector_coordinates = np.round(kmeans.cluster_centers_)
    reflector_coordinates = np.array([reflector_coordinates[0][1],reflector_coordinates[0][0]])
    return reflector_coordinates, reflector_segmentation
