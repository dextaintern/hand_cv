import cv2
import numpy as np
from math import atan2, degrees
from math import floor
from matplotlib import pyplot as plt
from matplotlib import cm

#video cap
cap = cv2.VideoCapture(0)

#skin colour bounds in hsv
lowbound = np.array([0, 0, 0])
uppbound = np.array([70, 255, 214])

def mapgrey(x):
    return cm.ScalarMappable(cmap='gray').to_rgba(x)[:,:,0]

def kern(n):
    return np.ones((n,n), np.uint8)

def findClusters(line):
    clusterongoing = False
    clusters = []; ccluster = []
    previdx = 0
    for idx,val in enumerate(line):
        if val == 0: continue
        if idx > (previdx + 20):
            #prev cluster ended
            clusterongoing = False

        if clusterongoing == False:
            #new cluster
            clusterongoing = True
            clusters += [ccluster]
            ccluster = []

        #agglomerate cluster
        ccluster += [idx]
        previdx = idx

    return clusters[1:]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape[:2]

#aid finger placement
    fingerbounds = [170,243,312,371,418]
    space = 15
    fingerrows = [(val+space, fingerbounds[i+1]-space) for i,val in enumerate(fingerbounds) if i < (len(fingerbounds)-1)]

    for l in fingerbounds:
        cv2.line(frame,(0,l),(cols,l),(0,0,0),thickness=3)

    for (s,e) in fingerrows:
        cv2.line(frame, (0, s), (cols, s), (30, 30, 30), thickness=1)
        cv2.line(frame, (0, e), (cols, e), (30, 30, 30), thickness=1)

    sobelx = cv2.Laplacian(img,cv2.CV_64F)#cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobelx = mapgrey(sobelx)
    sobelx = np.uint8(np.absolute(sobelx)*255)

#hand mask to remove hand edge
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # inrange only colours of hand, then remove gaps
    mask = cv2.inRange(hsv, lowbound, uppbound)
    mk = kern(5)
    mask = cv2.dilate(mask, mk, iterations=1)
    thinhandmask = cv2.erode(mask, mk, iterations=3)



#CANNY
    blur = sobelx#cv2.bilateralFilter(sobelx,5,70,70)#medianBlur(sobelx, 5)#
#    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    minval,maxval = 26,66
    '''    
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
            print( minval,maxval)
            break

        # edge detection
        edges = cv2.Canny(blur, minval, maxval, apertureSize=3, L2gradient=True)
        cv2.imshow('edges', edges)
    '''




    edges = cv2.Canny(blur, minval, maxval, apertureSize=3, L2gradient=True)

    # remove annoying fingertips
    edgesnotips = edges
    #edgesnotips = cv2.bitwise_and(edges, edges, mask=thinhandmask)
    edgesnotipscolour = cv2.cvtColor(edgesnotips, cv2.COLOR_GRAY2BGR)

#pretty
#    dilation = cv2.dilate(edgesnotips,kern(5),iterations = 2)##cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)#
#    dilation = cv2.erode(dilation,kern(5),iterations = 2)
    dilation=edgesnotips
    dilationcolour = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)



#find contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#eliminate contours
    for i,cnt in enumerate(contours):
        a = cv2.arcLength(cnt, closed=False)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(h) / w

        if a < 30 or a > 200:
            del contours[i]; cv2.drawContours(dilationcolour, cnt, -1, (0, 255, 255)); continue       #length condition, yellow
        if ratio <1:
            del contours[i]; cv2.drawContours(dilationcolour, cnt, -1, (255, 0, 255));  continue       #thinness condition, pink

        cv2.drawContours(dilationcolour, cnt, -1, (255, 0, 0))

#draw orientation lines
    for cnt in contours:
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)

        #cv2.line(dilationcolour, (cols - 1, righty), (0, lefty), (0, 255, 0), 1)
        llen = 20
        pt1 = (int(x - vx * llen), int(y - vy * llen))
        pt2 = (int(x + vx * llen), int(y + vy * llen))
        orientation = abs(degrees(atan2(pt2[0]-pt1[0],pt2[1]-pt1[1])))
        if orientation < 70:
            cv2.line(dilationcolour, pt1, pt2, (0, 255, 0))


    '''
    lines = cv2.HoughLines(dilation, 1, np.pi / 180, 60)
    print (lines[:,0])

    for rho, theta in lines[:,0]:
        if theta > 10*np.pi/180: continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(dilationcolour, (x1, y1), (x2, y2), (0, 0, 255), 2)
    '''


#isolate phalange division
    for fingerrow in fingerrows:

        (fingertippos,fingertiprow) = (9999999,9999999)
        phalanges = [[],[],[],[]]

        #scan finger width
        for yi in range(fingerrow[0], fingerrow[1], 5):
            fingline = np.flip(edgesnotips[yi])

            #draw clusters
            clusters = findClusters(fingline)
            if len(clusters) == 0: continue

            fingeredgepos = int(np.median(clusters[0]))
            if fingeredgepos < fingertippos:
                (fingertippos,fingertiprow) = (fingeredgepos, yi)

            for i,cluster in enumerate(clusters[:4]):
                clustavg = int(np.median(cluster))
                phalanges[i] += [clustavg]
                cv2.circle(frame,(cols-clustavg,yi),3,(0,0,255))

        #finger global points
        cv2.circle(frame, (cols - fingertippos, fingertiprow), 3, (255, 0, 0))

        for phalange in phalanges[1:]: #ignore tip
            if len(phalange) == 0: continue
            phalangeavg = int(np.median(phalange))
            cv2.circle(frame, (cols-phalangeavg, fingertiprow), 3, (255, 0, 0), -1)

    cv2.imshow('edges', edges)
    cv2.imshow('contours', dilationcolour)
    cv2.imshow('sobel', sobelx)
    cv2.imshow('lines no tips', edgesnotipscolour)
    cv2.imshow('frame',frame)
#    cv2.imshow('mask',thinhandmask)
#    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.waitKey(0)
#    cv2.waitKey(0)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()