import cv2
import os
import numpy as np
image = cv2.imread('21_training.tif')
kernel = np.ones((2,2), np.uint8)
kernel1 = np.ones((15,15), np.uint8)
kernel2 = np.ones((7,7), np.uint8)
imgcpy = image.copy()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# set blue and red channels to 0
imgcpy[:, :, 0] = 0
imgcpy[:, :, 2] = 0
img_grey = cv2.cvtColor(imgcpy, cv2.COLOR_BGR2GRAY)
img_grey=(255-img_grey)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
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
ret,xt = cv2.threshold(x,18,255,cv2.THRESH_TOZERO)
cv2.imshow("xt",xt)
x0 = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel1)
#cv2.imshow("op1",x0)
x1=cv2.subtract(x,x0)
cv2.imshow("x2",x1)
x10=cv2.morphologyEx(x1, cv2.MORPH_OPEN, kernel1)
x2=cv2.subtract(x1,x10)
cv2.imshow("x22",x2)
x3 =cv2.add(x1,x2)
cv2.imshow("x23",x3)
x20=cv2.morphologyEx(x2, cv2.MORPH_OPEN, kernel1)
x4=cv2.subtract(x3,x20)
x5=cv2.add(x2,x4)
x30=cv2.morphologyEx(x3, cv2.MORPH_OPEN, kernel1)
x6=cv2.subtract(x5,x30)
x40=cv2.morphologyEx(x4, cv2.MORPH_OPEN, kernel1)
x7=cv2.subtract(x6,x40)
cv2.imshow("x7",x7)
ret,thresh4 = cv2.threshold(x7,15,255,cv2.THRESH_TOZERO)
#x3 = cv2.morphologyEx(x2, cv2.MORPH_OPEN, kernel2)
#x4=cv2.subtract(x2,x3)
#x5=cv2.add(x2,x4)
#cv2.imshow("x5",x5)

mask = np.ones(x7.shape[:2], dtype="uint8") * 125	

im2, contours, hierarchy = cv2.findContours(x7.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	if cv2.contourArea(cnt) <= 5:
		cv2.drawContours(mask, [cnt], -1, 0, -1)			
im = cv2.bitwise_and(x7, x7, mask=mask)
ret,fin = cv2.threshold(im,15,255,cv2.THRESH_TOZERO)			
newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)), iterations=1)
res=cv2.add(newfin,newfin)
ret,thresh4 = cv2.threshold(x7,25,255,cv2.THRESH_BINARY)

cv2.imshow("new",thresh4)

cv2.waitKey(0)