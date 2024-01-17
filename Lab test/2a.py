import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("sun.jpeg", cv2.IMREAD_GRAYSCALE)
max_gray_level = 200
min_gray_level = 50
enhanced_image = np.copy(image)

enhanced_image[enhanced_image < min_gray_level] = min_gray_level
enhanced_image[enhanced_image > max_gray_level] = max_gray_level

plt.figure(figsize=(12,9))
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')

plt.subplot(1,3,2)
plt.title('Enhanced image')
plt.imshow(enhanced_image, cmap='gray')
plt.show()