import matplotlib.pyplot as plt
import numpy as np
import cv2

image_path = 'image.jpg'
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

new_image = image*0b00000111

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(new_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')
plt.show()