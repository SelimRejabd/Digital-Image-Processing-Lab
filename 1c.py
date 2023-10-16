import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('stone.jpg', cv2.IMREAD_GRAYSCALE)

histogram=[0]*256

for row in image:
    for pixel_value in row:
        histogram[int(pixel_value)] +=1

plt.figure(figsize=(12, 9))
plt.subplot(2, 2, 1)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.bar(range(256), histogram)
plt.xlim([0, 256])

plt.subplot(2, 2, 2)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

threshold_value = 128

thresholded_image=(image>=threshold_value).astype(int)

plt.subplot(2, 2, 4)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

histogram1 = [0] * 3

for row in thresholded_image:
    for pixel_value in row:
        histogram1[int(pixel_value)] += 1


plt.subplot(2, 2, 3)
plt.bar(range(3), histogram1)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('After Threshold Image Histogram')
plt.show()