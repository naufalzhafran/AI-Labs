import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('00000.jpg',1)
while True:
    cv2.imshow('image',img)
    cv2.destroyAllWindows()
# cv2.waitKey(0)