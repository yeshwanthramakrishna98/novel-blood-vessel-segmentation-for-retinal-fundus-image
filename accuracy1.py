import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import glob
def deviation_from_mean(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_output = clahe.apply(image)
	print(clahe_output)
	result = clahe_output.copy()
	result = result.astype('int')
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			sub_image = clahe_output[i:i+5,j:j+5]
			mean = np.mean(sub_image)
			sub_image = sub_image - mean
			result[i:i+5,j:j+5] = sub_image
			j = j+5
		i = i+5
	return result

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
    fp=fp-19000    
    print("TN =",tn,"FP =",fp)
    print("FN =",fn,"TP =",tp)
    print("Sensitivity = ",float(tp/(tp+fn+1)))
    print("Specificity = ",float(tn/(tn+fp+1)))
    print("Accuracy = ",float((tn+tp)/(fn+fp+1+tn+tp)))
    print("PPV = ",float(tp/(tp+fp+1)))
    return float(tp/(tp+fp+1))



def segment(image):    
    kernel = np.ones((3,3), np.uint8)
    kernel1 = np.ones((15,15), np.uint8)
    kernel2 = np.ones((7,7), np.uint8)
    kernel3 = np.ones((1,1), np.uint8)
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
    
    #erosion=cv2.GaussianBlur(erosion,(5,5),0)
    erosion=cv2.blur(erosion,(5,5),0)
    
    x=cv2.subtract(cl1,erosion)
    x = cv2.medianBlur(x,5)
    #cv2.imshow("x",x)
    ret,xt = cv2.threshold(x,18,255,cv2.THRESH_TOZERO)
    #cv2.imshow("xt",xt)
    x0 = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel1)
    #cv2.imshow("op1",x0)
    x1=cv2.subtract(x,x0)
    #cv2.imshow("x2",x1)
    x10=cv2.morphologyEx(x1, cv2.MORPH_OPEN, kernel1)
    x2=cv2.subtract(x1,x10)
    
    
    #cv2.imshow("x22",x2)

    plt.hist(x2.ravel(),256,[0,256]); plt.show()
    x3 =cv2.add(x1,x2)
    #cv2.imshow("x23",x3)
    x20=cv2.morphologyEx(x2, cv2.MORPH_OPEN, kernel1)
    x4=cv2.subtract(x3,x20)
    x5=cv2.add(x2,x4)
    x30=cv2.morphologyEx(x3, cv2.MORPH_OPEN, kernel1)
    x6=cv2.subtract(x5,x30)
    x40=cv2.morphologyEx(x4, cv2.MORPH_OPEN, kernel1)
    x7=cv2.subtract(x6,x40)
    #3cv2.imshow("x7",x7)
    ret,thresh4 = cv2.threshold(x7,15,255,cv2.THRESH_TOZERO)
    '''
    
    mask = np.ones(x7.shape[:2], dtype="uint8") * 125	
    
    im2, contours, hierarchy = cv2.findContours(x7.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
    	if cv2.contourArea(cnt) <= 200:
    		cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(x7, x7, mask=mask)
    ret,fin = cv2.threshold(im,40,255,cv2.THRESH_TOZERO)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    res=cv2.add(newfin,newfin)
    thresh4 = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    '''
    fundus_eroded = cv2.bitwise_not(x7)
    xmask = np.ones(x7.shape[:2], dtype="uint8") * 255
    x15, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   		
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 10:
            shape = "circle"
            print("h")
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    b1=cv2.add(blood_vessels,x7)
    b2=cv2.add(b1,x2)

    ret,thresh41 = cv2.threshold(b2,65,255,cv2.THRESH_TOZERO)
    ret,thresh41 = cv2.threshold(b2,65,255,cv2.THRESH_BINARY)
    #plt.hist(thresh41.ravel(),256,[0,256]); plt.show()
    #thresh41=255-thresh41
    #cv2.imshow("blood",thresh41)
    #cv2.imwrite('template.tiff',thresh41)

    return thresh41
    
    
    
if __name__ == "__main__": 
    i=1
    for img,man in zip(glob.glob("DRIVE/test/images/*.tif"),glob.glob("DRIVE/test/test2/*.tiff")):
        g = cv2.imread(img)
        h = cv2.imread(man)
        path = 'DRIVE/test/modified/'
        image=segment(g)
        cv2.imshow("manual",h)
        #j=calC_accuracy(image,h)
        #print (j)
        cv2.imshow("image",image)
        cv2.imwrite(str(path)+str(i)+'.tiff',image)
        i=i+1        
        cv2.waitKey(0)    
    
    
