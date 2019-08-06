import cv2

img = cv2.imread('sudoku.jpg',0)
img2 = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,3)

print(ret)

cv2.imshow('g', img)
cv2.imshow('h', th1)
cv2.waitKey(0)
cv2.destroyAllWindows()