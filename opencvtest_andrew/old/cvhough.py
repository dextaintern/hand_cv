import cv2
import numpy as np
import math

img = cv2.imread('sudoku.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

xmax, ymax,c = img.shape
rdim = math.floor(math.hypot(xmax, ymax))
thetadim = 180

thetamax = 1.0 * math.pi
thetamin = 0.0
print (img.shape)


hough_space = np.zeros((rdim,thetadim))

for x in range(xmax):
    for y in range(ymax):
        if edges[x,y] *1==0:
            continue
        for itheta in range(0,180):
            if itheta<0:itheta+=thetadim
            theta = 1.0 * itheta * thetamax / thetadim
            r = x * math.cos(theta) + y * math.sin(theta)
            #print(math.floor(r),math.floor(itheta))
            hough_space[math.floor(r),math.floor(itheta)] += 1

print(hough_space[100:200,100:200])

cv2.imshow('out', hough_space)
cv2.waitKey(0)
cv2.destroyAllWindows()