import cv2
import numpy as np
import matplotlib as plt
image_path = 'image.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Resized Image', cv2.WINDOW_NORMAL)
cv2.imshow('Resized Image', image)
cv2.waitKey(1000)
i = 0
while i<5:
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Resized Image', image)
    cv2.waitKey(1000)
    i = i + 1

cv2.destroyAllWindows()
