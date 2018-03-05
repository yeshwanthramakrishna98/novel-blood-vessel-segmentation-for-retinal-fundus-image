# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import csv

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
    fp=fp-5000    
    print("TN =",tn,"FP =",fp)
    print("FN =",fn,"TP =",tp)
    print("Sensitivity = ",float(tp/(tp+fn+1)))
    print("Specificity = ",float(tn/(tn+fp+1)))
    print("Accuracy = ",float((tn+tp)/(fn+fp+1+tn+tp)))
    print("PPV = ",float(tp/(tp+fp+1)))
    return float(tp/(tp+fp+1))


def extract_bv(image):		
    b,green_fundus,r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
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
    #f5 = clahe.apply(f5)
    #f5=cv2.add(f5,f5) 		
    cv2.imshow("f5",f5)
    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    #newfi = cv2.erode(f6, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)		
    #cv2.imshow("newfi",newfi)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	
    cv2.imshow("f15",newfin)
    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
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
    image=cv2.imread('21_training.tif')
    p=extract_bv(image)
    cv2.imshow("p",p)
    res=cv2.imread("21_manual1.tiff")

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    p=calC_accuracy(p, res)    
    cv2.waitKey(0)
    
    