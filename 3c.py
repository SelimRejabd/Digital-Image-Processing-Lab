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

def psnr_calculation(original_image, noisy_image):
    original_image = original_image.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    mse = np.mean((original_image-noisy_image)**2)
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return psnr

def harmonic_mean_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float64)

    # Calculate the kernel radius
    kernel_radius = kernel_size // 2

    for i in range(kernel_radius, height - kernel_radius):
        for j in range(kernel_radius, height - kernel_radius):
            values = []

            # Iterate over the neighborhood
            for x in range(i - kernel_radius, i + kernel_radius + 1):
                for y in range(j - kernel_radius, j + kernel_radius + 1):
                    if image[x, y] != 0:
                        values.append(1 / image[x, y])
            # Calculate the harmonic mean
            if values:
                filtered_image[i, j] = int(len(values) / np.sum(values))

    return filtered_image

kernel_size = 5
harmonic_filtered_image = harmonic_mean_filter(noisy_image, kernel_size)
psnr1 = psnr_calculation(image, noisy_image)
psnr2 = psnr_calculation(image, harmonic_filtered_image)


plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(noisy_image, cmap = 'gray')
plt.title(f'Noisy Image psnr: {psnr1:.2f}dB')

plt.subplot(1,3,3)
plt.imshow(harmonic_filtered_image, cmap = 'gray')
plt.title(f'Filtered Image psnr: {psnr2:.2f}dB')
plt.show()