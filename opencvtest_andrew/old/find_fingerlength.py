import cv2
import numpy as np
import math
import circle_fit as cf
def dist(x1,x2,y1,y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def midpoint(x1,x2,y1,y2):
    x,y = (x1+x2)/2,(y1+y2)/2
    return (x,y)
def find_angle(x1,x2,y1,y2):
    angle =math.atan2( abs(y2-y1),abs(x2-x1) )
    return angle


def process_img(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lowbound = np.array([0, 0, 0])
    uppbound = np.array([70, 255, 214])

    # inrange only colours of hand, then remove gaps
    mask = cv2.inRange(hsv, lowbound, uppbound)
    kernel = np.ones((5, 5), np.uint8)

    # dilate
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    hand = cv2.bitwise_and(frame,frame, mask=mask)

    print(hand.shape)
    im = hand.copy()
    hand=cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(hand,100,255,0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = contours
    hand_cnt = []
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # compute the rotated bounding box of the contour.
        x, y, w, h = cv2.boundingRect(c)
        if (x, y) == (0, 0):
            continue

        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(hand, c, -1, (0, 255, 255))
        hand_cnt.append(c)
    #keeping doing until find a hand
    if len(hand_cnt)>1 or len(hand_cnt)==0:
        cv2.imshow('hand',im)
        print("keep finding hand")
    else:
        c = hand_cnt[0]
        hull = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hull)

        # creat lists to store start points and far points
        start_p = []
        far_p = []
        points = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(c[s][0])
            end = tuple(c[e][0])

            far = tuple(c[f][0])
            start_p.append((c[s][0]))
            far_p.append(c[f][0])
            cv2.line(im, start, end, [0, 255, 0], 2)

            if i > 0:
                x1, y1 = far_p[i - 1]
                x2, y2 = far_p[i]
                if dist(x1, x2, y1, y2) > 50:
                    rightmost = tuple(c[c[:, :, 0].argmax()][0])
                    if abs(x2 - rightmost[0]) > 30:
                        points.append((x2, y2))
                else:
                    continue
        for i in range(len(points)):
            x, y = points[i]

            cv2.circle(im, (x, y), 5, [0, 0, 255], -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, str(i), (x + 10, y + 10), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # cv2.circle(im, start, 2, [0, 255, 255], -1)
        if len(points)!= 10:
            cv2.imshow('hand',im)
        else:
            even = [0, 2, 4, 6, 8, 10]
            odd = [1, 3, 5, 7, 9]
            fingers = {'pinky': 0, 'ring': 0, 'middle': 0, 'index': 0, 'thumb': 0}
            finger_length = []
            mid_points = []
            # find middle points, noted the one for index is inaccurate
            for i in range(len(even) - 1):
                x1, y1 = points[even[i]]
                x2, y2 = points[even[i + 1]]
                mx, my = midpoint(x1, x2, y1, y2)
                mid_points.append((mx, my))
                tx, ty = points[odd[i]]
                finger_length.append(dist(mx, tx, my, ty))

            # fit middle points in a circle curve
            data = []
            for i in range(4):
                data.append(points[even[i]])

            xc, yc, r, _ = cf.least_squares_circle(data)
            xc, yc = int(xc), int(yc)
            cv2.circle(im, (xc, yc), 5, [0, 255, 255], -1)
            mid_middle = mid_points[2]

            angle_mid_middle = find_angle(xc, mid_middle[0], yc, mid_middle[1])
            angle_6 = find_angle(xc, points[6][0], yc, points[6][1])
            # for the case centre is above point 6
            if yc >= mid_middle[1]:
                angle = (angle_6 - angle_mid_middle) + angle_6
            elif points[6][1] <= yc < mid_middle[1]:
                angle = (angle_6 + angle_mid_middle) + angle_6
            # correct the index finger length
            xi = xc - r * math.cos(angle)
            yi = yc - r * math.sin(angle)
            mid_index = (xi, yi)
            mid_points[3] = mid_index
            finger_length[3] = dist(points[7][0], xi, points[7][1], yi)
            for i in range(5):
                xm, ym = mid_points[i]
                xm, ym = int(xm), int(ym)
                cv2.circle(im, (xm, ym), 6, [175, 255, 255], -1)
            count = 0
            for name in fingers:
                fingers[name] = finger_length[count]
                count += 1
                print(name, ":", fingers[name])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(im, str(fingers[name]), (points[odd[count]][0] + 10, points[odd[count]][1] + 10), font, 0.5,
                            (255, 255, 175), 2, cv2.LINE_AA)
            cv2.imshow('frame', im)


cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    ret, frame = cap.read()
    process_img(frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()