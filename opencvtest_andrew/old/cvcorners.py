import numpy as np
import cv2


img = cv2.imread('box_in_scene.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
#corners = np.int0(corners)

for i in corners:

    cv2.circle(img,tuple(i[0]),3,255,-1)

cv2.imshow("h", img)
cv2.waitKey(0)
cv2.destroyAllWindows()