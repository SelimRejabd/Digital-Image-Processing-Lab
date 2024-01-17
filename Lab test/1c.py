import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("sun.jpeg", cv2.IMREAD_GRAYSCALE)

histogram = [0]*256
for row in image:
    for pixel in row:
        histogram[int(pixel)] += 1
threshold = 128

thresholded_image = (image >= threshold).astype(int)
histogram2 = [0]*5

for row in thresholded_image:
    for value in row:
        histogram2[int(value)] += 1

plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
plt.imshow(image, cmap='gray')
plt.subplot(2,2,2)
plt.bar(range(256), histogram)

plt.subplot(2,2,3)
plt.imshow(thresholded_image, cmap='gray')
plt.subplot(2,2,4)
plt.bar(range(5), histogram2)

plt.show()