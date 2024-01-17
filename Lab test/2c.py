import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("sun.jpeg", cv2.IMREAD_GRAYSCALE)
masked_image = image & 0b11100000

plt.figure(figsize=(12,9))
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')

plt.subplot(1,3,2)
plt.title('Masked image')
plt.imshow(masked_image, cmap='gray')
plt.show()