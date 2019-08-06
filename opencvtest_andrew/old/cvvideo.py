import cv2

cap = cv2.VideoCapture(0)
ret = cap.set(5,5.0)
print (cap.get(5))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    corners = cv2.goodFeaturesToTrack(gray, 50, 0.05, 20)
#    for i in corners:
#        cv2.circle(frame, tuple(i[0]), 3, 255, -1)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Display the resulting frame
    cv2.imshow('frame',edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()