import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

image = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)


def add_salt_noise(noisy_image):
    row, col = noisy_image.shape
    num_of_pixels = np.random.randint(300, 10000)
    for i in range(num_of_pixels):
        x_cor = np.random.randint(0, row-1)
        y_cor = np.random.randint(0, col-1)
        noisy_image[x_cor][y_cor] = 255
    return noisy_image


def add_papper_noise(noisy_image):
    row, col = noisy_image.shape
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


def harmonic_mean_filter(noisy_image, kernel_size):
    filtered_image = np.zeros_like(noisy_image)
    pad_size = kernel_size // 2
    row, col = noisy_image.shape
    for i in range(pad_size, row-pad_size):
        for j in range(pad_size, col-pad_size):
            window = noisy_image[i-pad_size:i +
                                 pad_size+1, j-pad_size:j+pad_size+1]
            non_zero_values = window[window != 0]
            non_zero_values = 1/non_zero_values
            harmonic_mean = (len(non_zero_values) / np.sum(non_zero_values))
            filtered_image[i][j] = harmonic_mean
    return filtered_image


def geometric_mean_filter(noisy_image, kernel_size):
    filtered_image = np.zeros_like(noisy_image)
    row, col = noisy_image.shape
    pad_size = kernel_size // 2
    for i in range(pad_size, row-pad_size):
        for j in range(pad_size, col-pad_size):
            window = noisy_image[i-pad_size:i +
                                 pad_size+1, j-pad_size: j+pad_size+1]
            non_zero_values = window[window != 0]

            if (len(non_zero_values) == 0):
                log_sum = np.sum(np.log(non_zero_values))
                geometric_mean = np.exp(log_sum/(kernel_size*kernel_size))
                filtered_image[i][j] = geometric_mean
            else:
                filtered_image[i][j] = np.median(window[window != 0])
    return filtered_image


harmonic_filtered_image = harmonic_mean_filter(noisy_image, 3)
noisy_image_psnr = psnr_calculation(image, noisy_image)
h_image_psnr = psnr_calculation(image, harmonic_filtered_image)
geometric_mean_filtered_image = geometric_mean_filter(noisy_image, 3)
g_image_psnr = psnr_calculation(image, geometric_mean_filtered_image)


plt.figure(figsize=(12, 4))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title(f'Noisy Image psnr: {noisy_image_psnr:.2f}dB')

plt.subplot(2, 2, 3)
plt.imshow(harmonic_filtered_image, cmap='gray')
plt.title(f'Filtered Image psnr: {h_image_psnr:.2f}dB')
plt.subplot(2, 2, 4)
plt.imshow(geometric_mean_filtered_image, cmap='gray')
plt.title(f'Geometric mean filtered image \n PSNR: {g_image_psnr:.2f}')
plt.tight_layout()
plt.show()
