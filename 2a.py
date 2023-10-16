import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('stone.jpg', cv2.IMREAD_GRAYSCALE)

min_gray_level = 50
max_gray_level = 200

enhanced_image = np.copy(image)
enhanced_image[enhanced_image < min_gray_level] = min_gray_level
enhanced_image[enhanced_image > max_gray_level] = max_gray_level

# enhanced_image = (255 * (enhanced_image - min_gray_level) / (max_gray_level - min_gray_level)).astype(np.uint8)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.show()
