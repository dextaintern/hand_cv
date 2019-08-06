import cv2
import numpy as np
from math import atan2, degrees, atan
from matplotlib import cm
import openpyxl
import find_fingerlength_edged_forlines3

BG = 'K'
FRAME_FLIP = True
SRC_VIDEO = False
SRC_FOLDER = "all_cali/"
SRC_IM = "chengjie_r.jpg"
R_HAND = True

gamma = 0.6;
alpha = 1.1;
beta = 0

n_lines = 3
n_fingers = 5

TEXT_SCALE = 0.4
FRAME_SCALE = 0.2
GREEN = (0,255,0); BLUE = (255,0,0); RED = (0,0,255); MAGENTA = (255,0,255); YELLOW = (0,255,255); CYAN = (255,255,0); WHITE = (255,255,255); BLACK = (0,0,0); GREY = (100,100,100)
savecount = 2

wbfn = 'saves/phalangedatacollec.xlsx'
wb = openpyxl.load_workbook(wbfn)
ws = wb['img']

fingspacer = 15
fingerbounds = [100,170,243,312,371,441][:n_fingers+1]
fingerbounds = [i-50 for i in fingerbounds]

#video cap
if SRC_VIDEO==True: cap= cv2.VideoCapture(0)

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def mapgrey(x):
    return cm.ScalarMappable(cmap='gray').to_rgba(x)[:,:,0]

def kern(n):
    return np.ones((n,n), np.uint8)

def findClusters(line, cluster_spacing=10):
    # find clusters of non-zero items in a 1D array
    # returns list of clusters of indices

    clusters = []; ccluster = []
    previdx = -cluster_spacing

    for idx,val in enumerate(line):
        if val == 0: continue
        if idx > (previdx + cluster_spacing):
            #prev cluster ended, so new cluster
            clusters += [ccluster]
            ccluster = []

        #agglomerate cluster
        ccluster += [idx]
        previdx = idx

    return clusters[1:]

def fingerRect(im, finger_key_points, rect_size = 15):
    side_cushion=15
    #rotate according to f_k_p
    #ROI of rect
    #return ROI
    rotangle = 90-degrees(atan2((finger_key_points[0][0]-finger_key_points[1][0]),(finger_key_points[0][1]-finger_key_points[1][1])))
    M = cv2.getRotationMatrix2D(finger_key_points[0], rotangle, 1) #ACW
    rot = cv2.warpAffine(im, M, (cols, rows))
    roi = rot[finger_key_points[0][1]-rect_size:finger_key_points[0][1]+rect_size,  side_cushion:cols]
    #rotc = cv2.cvtColor(rot, cv2.COLOR_GRAY2BGR)
    #cv2.circle(rotc, finger_key_points[0], 3, GREEN)

    return roi, roi.shape

while True:
# Capture
    if SRC_VIDEO == True:
        _, frame = cap.read()
    else:
        frame = cv2.imread(SRC_FOLDER+SRC_IM)
        frame = cv2.resize(frame,None,fx=FRAME_SCALE,fy=FRAME_SCALE)

    frame_copy = np.copy(frame)
    rows, cols = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = adjust_gamma(img,gamma)
    img = cv2.convertScaleAbs(img,alpha=alpha,beta=beta)

# Isolate hand from shadow
    if BG == 'W':
        lowbound = np.array([0, 0, 0])
        uppbound = np.array([70, 255, 214])
    elif BG == 'K':
        lowbound = np.array([0, 0, 50])
        uppbound = np.array([179, 255, 255])

# inrange only colours of hand, then remove gaps
    mask = cv2.inRange(hsv, lowbound, uppbound)
    mask = cv2.dilate(mask, kern(5), iterations=1)
    mask = cv2.erode(mask, kern(5), iterations=1)

# Locate fingers
    fingerrows = [(val + fingspacer, fingerbounds[i + 1] - fingspacer) for i, val in enumerate(fingerbounds) if i < (len(fingerbounds) - 1)]

    img = cv2.bitwise_and(img, img, mask=mask)

# Make finger image
    if SRC_VIDEO == False:
        img_created = np.zeros(img.shape, np.uint8)
        hand_finger_key_points,hand_finger_lengths = find_fingerlength_edged_forlines3.process_img(frame_copy, right_hand=R_HAND)
        hand_finger_lengths = list(reversed(hand_finger_lengths))
        for i,finger_key_points in enumerate(list(reversed(hand_finger_key_points))[:n_fingers]):
            rect,s = fingerRect(img, finger_key_points)
            cv2.circle(frame, finger_key_points[0], 7, GREEN)
            cv2.circle(frame, finger_key_points[1], 7, YELLOW)
            img_created[fingerbounds[i]+fingspacer:fingerbounds[i]+fingspacer+s[0]  ,  0:s[1]] = rect
        img_created = img_created[:, 0:s[1]]
        img_created = cv2.flip(img_created, 1)

        rows, cols = img_created.shape[:2]

    else:
        img_created = img


# SOBEL X Filter and map colours
    sobelx = cv2.Sobel(img_created, cv2.CV_64F, 1, 0, ksize=5)#cv2.Laplacian(img,cv2.CV_64F)
    sobelx = mapgrey(sobelx)
    sobelx = np.uint8(np.absolute(sobelx)*255)

    #sobelx = cv2.bitwise_and(sobelx, sobelx, mask=mask)

# Blur Filter
    blur = sobelx#cv2.bilateralFilter(sobelx,5,70,70)#medianBlur(sobelx, 5)
    

# CANNY Edge detection
    canny_min,canny_max = 26,66

    '''    
    while True:
        k = cv2.waitKey(2) & 0xFF
        if k == ord('a'):
            canny_min = canny_min - 1 if canny_min != 0 else 0
        elif k == ord('d'):
            canny_min = canny_min + 1 if canny_min < canny_max else canny_min
        elif k == ord('j'):
            canny_max = canny_max - 1 if canny_min < canny_max else canny_max
        elif k == ord('l'):
            canny_max = canny_max + 1 if canny_max != 255 else 255
        elif k == 27:
            print( canny_min,canny_max)
            break

        # edge detection
        edges = cv2.Canny(blur, canny_min, canny_max, apertureSize=3, L2gradient=True)
        cv2.imshow('edges', edges)
    '''

    edges = cv2.Canny(blur, canny_min, canny_max, apertureSize=3, L2gradient=True)

# Morph to remove artifacts
    dilation = edges
    img_bin = dilation


# Scan fingers and Isolate phalange divisions
    draw_img = cv2.cvtColor(img_created, cv2.COLOR_GRAY2BGR)

    phalange_separations = [[] for i in range(n_fingers)]
    ratios = []

    for fi, fingerrow in enumerate(fingerrows):

        fingertipx_global,fingertipy_global = 9999999,9999999
        phalanges = [[] for i in range(n_lines)]

        # scan finger width
        for yi in range(fingerrow[0], fingerrow[1], 1):
            fingline = np.flip(img_bin[yi])

            # draw clusters
            clusters = findClusters(fingline)
            if len(clusters) == 0: continue

            # find finger tips
            fingertipx = int(np.median(clusters[0]))
            if fingertipx_global - fingertipx > 0:
                (fingertipx_global,fingertipy_global) = (fingertipx, yi)

            # save phalange positions
            for i,cluster in enumerate(clusters[:n_lines]):
                clustavg = int(np.median(cluster))
                phalanges[i] += [clustavg]
                cv2.circle(draw_img,(cols-clustavg,yi),3,RED)

        # Save finger global positions
        #tip
        phalange_pos = [fingertipx_global]
        disp_row = int(0.5 * (fingerrow[1] + fingerrow[0]))
        cv2.circle(draw_img, (cols - fingertipx_global, disp_row), 3, GREEN)
        for i,phalange in enumerate(phalanges[1:]): #ignore tip
            if len(phalange) == 0: continue
            phalangeavg = int(np.median(phalange))

            #if i==0: #tip
            #    phalangeavg += int(np.subtract(*np.percentile(phalange, [75, 25]))) #Interquartile Range

            phalange_pos += [phalangeavg]
            cv2.circle(draw_img, (cols - phalangeavg, disp_row), 3, GREEN, -1)

        #calculate separations
        for i,pos in enumerate(phalange_pos):
            if i == len(phalange_pos)-1: break
            separation = phalange_pos[i+1] - pos
            phalange_separations[fi] += [separation]

            cv2.putText(draw_img, str(separation), (cols-int(pos + separation/2),fingertipy_global), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, GREEN, 1, cv2.LINE_AA)


#Display Results

# (draw finger bounds)
    for l in fingerbounds:
        cv2.line(draw_img,(0,l),(cols,l),BLUE,thickness=3)

    for (s,e) in fingerrows:
        cv2.line(draw_img, (0, s), (cols, s), BLUE, thickness=1)
        cv2.line(draw_img, (0, e), (cols, e), BLUE, thickness=1)

# Display images
    cv2.imshow('mask', mask)
    cv2.imshow('edges', img_bin)
    cv2.imshow('x-gradient', sobelx)
    cv2.imshow('frame',frame)
    cv2.imshow('img',img)
    cv2.imshow('lined up',img_created)
    cv2.imshow('draw_img', draw_img)

#    cv2.waitKey(0)
    k = cv2.waitKey(1) & 0xFF
    step = 10
    if k == ord(' '):
        cv2.waitKey(0)
    elif k == ord('z'):
        print("saving as edges" + str(savecount))
        cv2.imwrite('saves/edges{}.jpg'.format(savecount), edges)
        cv2.imwrite('saves/x-gradient{}.jpg'.format(savecount), sobelx)
        cv2.imwrite('saves/frame{}.jpg'.format(savecount), frame)
        savecount += 1
    elif k == ord('x'):
        try:
            print("saving to excel, "+SRC_IM)
            print(phalange_separations)
            for i,fing in enumerate(phalange_separations[:1]): #just thumb
                ws.append([SRC_IM]+[hand_finger_lengths[i]]+fing+[sum(fing)]+[alpha,beta,gamma])
            wb.save(wbfn)
        except PermissionError:
            print("saving permission denied")
    elif k == ord(']') and np.max(fingerbounds) < rows:
        fingerbounds = [int(i*1.05) for i in fingerbounds ]
    elif k == ord('['):
        fingerbounds = [int(i/1.05) for i in fingerbounds]
    elif k == ord('1'):
        fingerbounds[0] -= step
    elif k == ord('2'):
        fingerbounds[0] += step
    elif k == ord('3'):
        fingerbounds[1] -= step
    elif k == ord('4'):
        fingerbounds[1] += step
    elif k == ord('5'):
        fingerbounds[2] -= step
    elif k == ord('6'):
        fingerbounds[2] += step
    elif k == ord('7'):
        fingerbounds[3] -= step
    elif k == ord('8'):
        fingerbounds[3] += step
    elif k == ord('9'):
        fingerbounds[4] -= step
    elif k == ord('0'):
        fingerbounds[4] += step
    elif k == ord('-'):
        fingerbounds[5] -= step
    elif k == ord('='):
        fingerbounds[5] += step
    elif k == ord('a'):
        alpha -= 0.1
        print("alpha,beta,gamma=",str(alpha),str(beta),str(gamma))
    elif k == ord('s'):
        alpha += 0.1
        print("alpha,beta,gamma=",str(alpha),str(beta),str(gamma))
    elif k == ord('g'):
        gamma -= 0.1
        print("alpha,beta,gamma=",str(alpha),str(beta),str(gamma))
    elif k == ord('h'):
        gamma += 0.1
        print("alpha,beta,gamma=",str(alpha),str(beta),str(gamma))
    elif k == ord('b'):
        beta -= 1
        print("alpha,beta,gamma=",str(alpha),str(beta),str(gamma))
    elif k == ord('n'):
        beta += 1
        print("alpha,beta,gamma=",str(alpha),str(beta),str(gamma))
    elif k == 27:
        break


# When everything done, release the capture
print("Terminating...")
cap.release()
cv2.destroyAllWindows()