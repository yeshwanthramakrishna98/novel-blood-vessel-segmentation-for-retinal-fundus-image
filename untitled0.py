# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:19:09 2018

@author: yeshwanth R
"""

import cv2
import numpy as np
import os
import csv

def accuracy_check(result, label):
    a = 0
    b = 0
    c = 0
    d = 0
    i = 0
    j = 0

    print(np.unique(result))
    print(np.unique(label))

    while i < result.shape[0]:
    	j = 0
    	while j < result.shape[1]:
    		if label[i,j] == 255:
    			if result[i,j] == label[i,j]:
    				a = a + 1
    			else:
    				d = d + 1
    		else:
    			if result[i,j] == label[i,j]:
    				c = c + 1
    			else:
    				b = b + 1
    		j = j + 1
    	i = i + 1
       
  #  print("TN =",tn,"FP =",fp)
  #  print("FN =",fn,"TP =",tp)
    print("Sensitivity of the image = ",float(a/(a+d+1)))
    print("Specificity of the image = ",float(c/(c+b+1)))
    print("Accuracy of the image = ",float((c+a)/(d+b+1+c+a)))
    print("PPV = ",float(a/(a+b+1)))
    return float(a/(a+b+1))


def extract_ab(image):		
    image[:, :, 0] = 0
    image[:, :, 2] = 0
    green_extract = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    contrast_enhanced_green_fundus = clahe.apply(green_extract)

    
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    r4 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R4 = cv2.morphologyEx(r4, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    f4 = cv2.subtract(R4,contrast_enhanced_green_fundus)
    
    f5 = clahe.apply(f4)
    
    cv2.imshow("f5",f5)
    
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)), iterations=1)	
    cv2.imshow("f15",newfin)
    
    fundus_eroded = cv2.bitwise_not(newfin)	
    xmask = np.ones(f5.shape[:2], dtype="uint8") * 1
    x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 10:
            shape = "circle"	
            #print("h")     
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    blood_vessels = cv2.bitwise_not(finimage);blood_vessels=255-blood_vessels 
    return blood_vessels	

if __name__ == "__main__":	
    image=cv2.imread('02_test.tif')
    p=extract_ab(image)
    cv2.imshow("p",p)
    res=cv2.imread("02_manual1.tiff")

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    print(res.shape)
    p=accuracy_check(p, res)    
    cv2.waitKey(0)