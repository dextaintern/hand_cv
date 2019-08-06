import cv2
import numpy as np

cap = cv2.VideoCapture(0)
lsat = 31
hsat = 219


while(1):

    # Take each frame
    ret, frame = cap.read()


#GET THIN FINGERS:

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lowbound = np.array([0, 0, 0])
    uppbound = np.array([70, 255, 214])

    # inrange only colours of hand, then remove gaps
    mask = cv2.inRange(hsv, lowbound, uppbound)
    kernel = np.ones((5,5), np.uint8)

#dilate
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=3)


    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        lsat = lsat - 1 if lsat != 0 else 0
        print(lsat,hsat)
    elif k == ord('d'):
        lsat = lsat + 1 if lsat < hsat else lsat
        print(lsat, hsat)
    elif k == ord('j'):
        hsat = hsat - 1 if lsat < hsat else hsat
        print(lsat, hsat)
    elif k == ord('l'):
        hsat = hsat + 1 if hsat != 255 else 255
        print(lsat, hsat)

cv2.destroyAllWindows()