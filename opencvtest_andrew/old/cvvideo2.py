import cv2
import numpy as np
from math import floor
from matplotlib import pyplot as plt

#video cap
cap = cv2.VideoCapture(0)
print( cap.set(cv2.CAP_PROP_FPS,5.0))
print (cap.get(cv2.CAP_PROP_FPS))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #apply fourier
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    cv2.imshow('img', img)

    rows, cols = img.shape
    crow, ccol = floor(rows / 2), floor(cols / 2)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.ones((rows, cols, 2), np.uint8)
    winsize=20
    mask[crow - winsize:crow + winsize, ccol - winsize:ccol + winsize] = 0


    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.imshow(img_back, cmap='gray')
    plt.show()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()