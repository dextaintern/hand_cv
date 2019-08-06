import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('digitalocean-logo.png',1)
cvs = np.zeros((512,512,3) ,np.uint8)
cvs = cv2.line(cvs,(0,0),(511,511),(255,0,0),5)
cvs = cv2.rectangle(cvs,(384,0),(510,128),(0,255,0),3)

cv2.imshow('im2',cvs)
cv2.waitKey(0)

#plt.imshow(img, interpolation = 'bicubic')
#plt.show()