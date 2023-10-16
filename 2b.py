import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sun.jpeg', cv2.IMREAD_GRAYSCALE)

gamma = 2.0

c = 1.0

power_law_transformed = c * np.power(image, gamma)

inverse_logarithm_image = c * np.exp(image/255)
# logarithm_image = c * np.log(1+image)
inverse_logarithm_image = np.uint8(inverse_logarithm_image)

difference_image = power_law_transformed - inverse_logarithm_image

plt.figure(figsize=(12, 4))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(power_law_transformed, cmap='gray')
plt.title('Power-Law Transformed')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(inverse_logarithm_image, cmap='gray')
plt.title('Inverse Logarithmic Transformed')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(difference_image, cmap='gray')
plt.title('Difference image')
plt.axis('off')


plt.tight_layout()
plt.show()
