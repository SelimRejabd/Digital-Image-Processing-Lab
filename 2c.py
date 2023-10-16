import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread('sun.jpeg', cv2.IMREAD_GRAYSCALE)

last_three_bits_image = original_image & 0b11100000

difference_image = original_image - last_three_bits_image

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(last_three_bits_image, cmap='gray')
plt.title('Last Three Bits (MSB)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(difference_image, cmap='gray')
plt.title('Difference Image')
plt.axis('off')

plt.tight_layout()
plt.show()
