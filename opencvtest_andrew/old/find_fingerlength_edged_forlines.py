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
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur it
    img = cv2.GaussianBlur(img, (7, 7), 0)

    ret, thresh = cv2.threshold(img, 70, 255, 0)
    img = cv2.bitwise_and(img, thresh)
    minval = 0
    maxval = 255


    # ret,edg = cv2.threshold(img,12,255,cv2.THRESH_BINARY)
    # cv2.imshow('i', edg)

    # edge detection
    edg = cv2.Canny(img, minval, maxval, apertureSize=3, L2gradient=True)
    # countour detection
    # minval,maxval = 84,84
    # opening and closing
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(edg, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(edg, cv2.MORPH_CLOSE, kernel)
    edg = closing


    contours, hierarchy = cv2.findContours(edg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hand = cv2.bitwise_and(frame,frame, mask=thresh)

    #print(hand.shape)
    im = hand.copy()

    cnts = contours
    hand_cnt = []
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 400:
            continue

        # compute the rotated bounding box of the contour.
        x, y, w, h = cv2.boundingRect(c)
        if (x, y) == (0, 0):
            continue

        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(im, c, -1, (0, 255, 255))
        hand_cnt.append(c)
    #keeping doing until find a hand
    if len(hand_cnt)!=1:
        cv2.imshow('hand',im)

    else:
        c = hand_cnt[0]
        hull = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hull)

        # creat lists to store start points and far points
        start_p = []
        end_p =[]
        far_p = []
        points = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(c[s][0])
            end = tuple(c[e][0])

            far = tuple(c[f][0])
            start_p.append((c[s][0]))
            end_p.append((c[e][0]))
            far_p.append(c[f][0])
            cv2.line(im, start, end, [0, 255, 0], 2)

            if i > 0:
                #defect points
                x1, y1 = far_p[i - 1]
                x2, y2 = far_p[i]
                #start and end points
                sx, sy = start_p[i]
                sx_1, sy_1 = start_p[i - 1]
                ex, ey = end_p[i]
                #take the only start point at finger tips
                if dist(sx, sx_1, sy, sy_1) > 40:
                    if dist(sx, ex, sy, ey) > 40:
                        points.append((sx, sy))

                if dist(x1, x2, y1, y2) > 35:
                    rightmost = tuple(c[c[:, :, 0].argmax()][0])
                    if abs(x2 - rightmost[0]) > 80:

                        points.append((x2, y2))
                else:
                    continue
        pc = []

        for i in range(len(points)):
            px,py = points[i]
            if abs(px - rightmost[0]) >70:
                pc.append((px, py))
        points =pc

        for i in range(len(points)):
            x, y = points[i]

            cv2.circle(im, (x, y), 5, [0, 0, 255], -1)
            #plot start points
            sx,sy =start_p[i]
            #cv2.circle(im, (sx, sy), 3, [0, 125, 255], -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, str(i), (x + 10, y + 10), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # cv2.circle(im, start, 2, [0, 255, 255], -1)


        if len(points)<10:
            print("not enough contour points")
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
            #get points for line construction for each finger
            finger_line = []
            for i in range(4):
                a1, b1 = points[odd[i]]
                a1,b1 = int(a1), int(b1)
                a2, b2 = mid_points[i]
                a2,b2 = int(a2), int(b2)
                finger_line.append([(a2,b2),(a1,b1)])
            angle_mid_middle = find_angle(xc, mid_middle[0], yc, mid_middle[1])
            angle_6 = find_angle(xc, points[6][0], yc, points[6][1])
            # for the case centre is below point 6
            if yc >= mid_middle[1]:
                angle = (angle_6 - angle_mid_middle) + angle_6
                xi = xc - r * math.cos(angle)
                yi = yc - r * math.sin(angle)
            #for the case yc is in between
            elif points[6][1] <= yc < mid_middle[1]:
                angle = (angle_6 + angle_mid_middle) + angle_6
                xi = xc - r * math.cos(angle)
                yi = yc - r * math.sin(angle)
            else:
                #for the case points 6 and mid_middle both below yc
                if (angle_mid_middle - angle_6) <= angle_6:
                    angle = angle_6 - (angle_mid_middle - angle_6)
                    xi = xc - r * math.cos(angle)
                    yi = yc + r * math.sin(angle)
                #for the case index_mid will be above yc
                else:
                    angle = (angle_mid_middle - angle_6) - angle_6
                    xi = xc - r * math.cos(angle)
                    yi = yc - r * math.sin(angle)

            # correct the index finger length
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
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(im, str(int(fingers[name])), (points[odd[count]][0] - 10, points[odd[count]][1] - 10), font,
                            0.5,(255, 255, 175), 2, cv2.LINE_AA)
                count += 1
            return finger_line

