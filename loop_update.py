# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 19:35:11 2018

@author: yeshwanth R
"""

import cv2
import numpy as np
import os
import csv
import glob
def calC_accuracy(result, label):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    i = 0
    j = 0

    print(np.unique(result))
    print(np.unique(label))

    while i < result.shape[0]:
    	j = 0
    	while j < result.shape[1]:
    		if label[i,j] == 255:
    			if result[i,j] == label[i,j]:
    				tp = tp + 1
    			else:
    				fn = fn + 1
    		else:
    			if result[i,j] == label[i,j]:
    				tn = tn + 1
    			else:
    				fp = fp + 1
    		j = j + 1
    	i = i + 1
       
    print("TN =",tn,"FP =",fp)
    print("FN =",fn,"TP =",tp)
    print("Sensitivity = ",float(tp/(tp+fn+1)))
    print("Specificity = ",float(tn/(tn+fp+1)))
    print("Accuracy = ",float((tn+tp)/(fn+fp+1+tn+tp)))
    print("PPV = ",float(tp/(tp+fp+1)))
    return float(tp/(tp+fp+1))


def extract_image(image):		
    image[:, :, 0] = 0
    image[:, :, 2] = 0
    green_extractimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    enhanced_greenimg = clahe.apply(green_extractimg)

    
    r1_opn = cv2.morphologyEx(enhanced_greenimg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r1_cls = cv2.morphologyEx(r1_opn, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2_opn = cv2.morphologyEx(r1_cls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r2_cls = cv2.morphologyEx(r2_opn, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3_opn = cv2.morphologyEx(r2_cls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    r3_cls = cv2.morphologyEx(r3_opn, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    r4_opn = cv2.morphologyEx(r3_cls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    r4_cls = cv2.morphologyEx(r4_opn, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    after_morpho_sub = cv2.subtract(r4_cls,enhanced_greenimg)
    
    f5 = clahe.apply(after_morpho_sub)
    
    cv2.imshow("image_after_morphology",f5)
    
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)			
    img_bitop = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(img_bitop,15,255,cv2.THRESH_BINARY_INV)			
    erod_img = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)), iterations=1)	
    cv2.imshow("f15",erod_img)
    
    fundus_eroded = cv2.bitwise_not(erod_img)	
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
  i=1
  for img,man in zip(glob.glob("DRIVE/test/images/*.tif"),glob.glob("DRIVE/test/test1/*.tiff")):
    image=cv2.imread(img)
    p=extract_image(image)
    h, w = p.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
 
    
    cv2.floodFill(p, mask, (270,2), 0);

    cv2.imshow("p",p)
    path = 'DRIVE/test/modified/'
    cv2.imwrite(str(path)+str(i)+'.tiff',p)
    res=cv2.imread(man)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    print(res.shape)
    i=i+1
    p=calC_accuracy(p, res)    
    cv2.waitKey(0)
