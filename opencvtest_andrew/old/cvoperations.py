import cv2
img = cv2.imread('messi.jpg')

img[:,:,2]=0

img= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])

cv2.imshow('img',img)
print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()