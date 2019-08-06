import cv2
import numpy as np

im = cv2.imread("opencv_logo.png")
im = cv2.resize(im, None, fx=0.5,fy=0.5)
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


minval = 0
maxval = 255


#ret,edg = cv2.threshold(img,12,255,cv2.THRESH_BINARY)
#cv2.imshow('i', edg)

while True:
    k = cv2.waitKey(2) & 0xFF
    if k == ord('a'):
        minval = minval - 1 if minval != 0 else 0
    elif k == ord('d'):
        minval = minval + 1 if minval < maxval else minval
    elif k == ord('j'):
        maxval = maxval - 1 if minval < maxval else maxval
    elif k == ord('l'):
        maxval = maxval + 1 if maxval != 255 else 255
    elif k == 27:
        break

    #edge detection
    edg = cv2.Canny(img, minval, maxval, apertureSize=3, L2gradient=True)
    cv2.imshow('i', edg)


#countour detection
contours, hierarchy = cv2.findContours(edg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

cv2.drawContours(im, contours, -1, (0,255,255))

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),5)

M = cv2.moments(cnt)
cv2.circle(im, (int(M['m10']/M['m00']), int(M['m01']/M['m00'])), 5, (0,0,255),-1)

hull = cv2.convexHull(cnt)
for pt in hull:
    cv2.circle(im, tuple(pt.tolist()[0]), 3, (255,0,0), -1)

print (cv2.contourArea(cnt)/cv2.contourArea(hull))

print (cv2.matchShapes(contours[0],contours[1],1,0.0))

cv2.imshow('j',im)
cv2.waitKey(0)
cv2.destroyAllWindows()