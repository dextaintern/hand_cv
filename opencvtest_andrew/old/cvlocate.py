import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# mouse callback function
def locate(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(x,y)


cv2.namedWindow('image')
cv2.setMouseCallback('image',locate)

while(1):
    ret, frame = cap.read()
    cv2.imshow('image',frame)
    cv2.waitKey(1)
