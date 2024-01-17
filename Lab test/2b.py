import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("sun.jpeg", cv2.IMREAD_GRAYSCALE)

gamma = 2.0
c = 1
power_law_transformed = c * np.power(image, gamma)

inverse_logarithmic_image = c * np.exp(image/255)

plt.figure(figsize=(12,9))
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')

plt.subplot(1,3,2)
plt.title('Power law transformed')
plt.imshow(power_law_transformed, cmap='gray')

plt.subplot(1,3,3)
plt.title('Inverse logarithmic image')
plt.imshow(inverse_logarithmic_image, cmap='gray')
plt.show()