import cv2
import numpy as np

img = cv2.imread('opencv_logo.png')
img = cv2.resize(img, None, fx=0.2,fy=0.2)

kernel = np.array([[3,1,1,1,1],[1,1,1,1,1],[1,1,1,1,3]])/29

dst = cv2.filter2D(img,-1,kernel)

cv2.imshow('h', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()