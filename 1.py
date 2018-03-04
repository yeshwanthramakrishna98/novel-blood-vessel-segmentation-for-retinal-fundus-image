import cv2
import glob
import numpy as np
kernel = np.ones((2,2), np.uint8)
i=1
for img in glob.glob("DRIVE/training/images/*.tif"):
    g = cv2.imread(img)
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    cv2.imshow("op1",g)
    #do process ,storing starts here
    path = 'DRIVE/training/modified/'
    image=g
    cv2.imwrite(str(path)+str(i)+'.tiff',image)
    i=i+1
    #till this.
    cv2.waitKey(0)
cv2.destroyAllWindows()
