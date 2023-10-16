import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

image = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)


def add_salt_noise(noisy_image):
    row,col = noisy_image.shape
    num_of_pixels = np.random.randint(300, 10000)
    for i in range(num_of_pixels):
        x_cor = np.random.randint(0, row-1)
        y_cor = np.random.randint(0, col-1)
        noisy_image[x_cor][y_cor] = 255
    return noisy_image
def add_papper_noise(noisy_image):
    row,col = noisy_image.shape
    num_of_pixels = np.random.randint(300, 10000)
    for i in range(num_of_pixels):
        x_cor = np.random.randint(0, row-1)
        y_cor = np.random.randint(0, col-1)
        noisy_image[x_cor][y_cor] = 0
    return noisy_image

noisy_image = image.copy()
add_salt_noise(noisy_image)
add_papper_noise(noisy_image)

# def harmonic_filter(noisy_image, x, y, a, b, total_pixels):
#     result = 0

#     for i in range(-a, a) :
#         for j in range(-b, b):
#             if image[x+i, y+j]>0:
#                 result += 1/image[x+i, y+j]
    
#     return total_pixels/result
neighborhood_size = 5
filtered_image = cv2.filter2D(noisy_image, -1, np.ones((neighborhood_size, neighborhood_size), dtype=np.float32) / (1.0 * neighborhood_size * neighborhood_size))



plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(noisy_image, cmap = 'gray')
plt.title('Noisy Image')

plt.subplot(1,3,3)
plt.imshow(filtered_image, cmap = 'gray')
plt.title('Filtered Image')
plt.show()