import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

image = cv2.imread("moon.jpg", cv2.IMREAD_GRAYSCALE)

def add_papper_soise(noisy_image):
    num_of_pixels = np.random.randint(3000, 10000)
    for i in range(num_of_pixels):
        [row, col] = image.shape
        x_cor = np.random.randint(0, row-1)
        y_cor = np.random.randint(0, col-1)
        noisy_image[x_cor][y_cor] = 255
    return noisy_image

def add_salt_soise(noisy_image):
    num_of_pixels = np.random.randint(3000, 10000)
    for i in range(num_of_pixels):
        [row, col] = image.shape
        x_cor = np.random.randint(0, row-1)
        y_cor = np.random.randint(0, col-1)
        noisy_image[x_cor][y_cor] = 0
    return noisy_image

noisy_image = image.copy()
add_salt_soise(noisy_image)
add_papper_soise(noisy_image)

def psnr_calculation(original_image, noisy_image):
    original_image = original_image.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    mse = np.mean((original_image-noisy_image)**2)
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return psnr

def average_filter(noisy_image, kernel_size):
    filtered_image = np.zeros_like(noisy_image)
    pad_size = kernel_size // 2
    starting_row = 0 + pad_size
    starting_col = 0 + pad_size
    ending_row = filtered_image.shape[0] - pad_size
    ending_col = filtered_image.shape[1] - pad_size
    mask = np.ones((kernel_size, kernel_size) / (kernel_size ** 2))

    for i in range(starting_row, ending_row):
        for j in range(starting_col, ending_col):
            window = noisy_image[i-pad_size : i+pad_size+1, j-pad_size : j + pad_size + 1]
            temp_window = window.copy()
            temp_window = temp_window * mask
            temp_window_mean = np.sum(temp_window)
            filtered_image[i][j] = temp_window_mean
    return filtered_image

def median_filter(noisy_image, kernel_size):
    filter_image = np.zeros_like(noisy_image)
    pad_size = kernel_size // 2
    starting_row = 0+pad_size
    starting_col = 0+pad_size
    ending_row = filter_image.shape[0] + pad_size
    ending_col = filter_image.shape[1] + pad_size

    for i in range(starting_row, ending_row):
        for j in range(starting_col, ending_col):
            window = noisy_image[i-pad_size : i+pad_size+1, j-pad_size : j+pad_size+1]
            window_median = np.median(window)
            filter_image[i][j] = window_median
    return filter_image 


kernel_size = 5
psnr = psnr_calculation(image, noisy_image)
psnr2 = peak_signal_noise_ratio(image, noisy_image)
plt.imshow(noisy_image, cmap='gray')
plt.title(f'{psnr:.2f}dB & {psnr2:.2f} dB' )
plt.show()
