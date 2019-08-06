import cv2
frame_in = cv2.imread("tester/"+'ruimin_r_nb.jpg')
frame_in = cv2.resize(frame_in, None, fx=0.2,fy=0.2)
rows,cols,c = frame_in.shape
print(rows,cols)