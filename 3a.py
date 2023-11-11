import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt


image = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)


def add_salt_noise(noise_image):
    row,col = noise_image.shape
    num_of_pixels = np.random.randint(300, 10000)
    for i in range(num_of_pixels):
        x_cor = np.random.randint(0, row-1)
        y_cor = np.random.randint(0, col-1)
        noise_image[x_cor][y_cor] = 255
    return noise_image
def add_papper_noise(noise_image):
    row,col = noise_image.shape
    num_of_pixels = np.random.randint(300, 10000)
    for i in range(num_of_pixels):
        x_cor = np.random.randint(0, row-1)
        y_cor = np.random.randint(0, col-1)
        noise_image[x_cor][y_cor] = 0
    return noise_image

noise_image = image.copy()
add_salt_noise(noise_image)
add_papper_noise(noise_image)

def calculate_psnr(original_image, noisy_image):
    original_image = original_image.astype(np.float64)
    noisy_image = noisy_image.astype(np.float64)
    mse = np.mean((original_image - noisy_image) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def average_filter(noisy_image, kernel_size):
    filtered_image = np.zeros_like(noisy_image)
    pad_size = kernel_size // 2
    starting_row = 0+pad_size
    starting_col = 0+pad_size
    ending_row = filtered_image.shape[0] - pad_size
    ending_col = filtered_image.shape[1] - pad_size
    mask = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    for i in range(starting_row, ending_row):
        for j in range(starting_col, ending_col):
            window = noisy_image[i - pad_size:i + pad_size + 1 , j-pad_size:j+pad_size+1]
            temp_window = window.copy()
            temp_window = temp_window * mask
            temp_window_mean = np.sum(temp_window)
            filtered_image[i][j] = temp_window_mean

    return filtered_image

def median_filter(noisy_image, kernel_size):
    filtered_image = np.zeros_like(noisy_image)
    pad_size = kernel_size // 2
    starting_row = 0+pad_size
    starting_col = 0+pad_size
    ending_row = filtered_image.shape[0] - pad_size
    ending_col = filtered_image.shape[1] - pad_size

    for i in range(starting_row, ending_row):
        for j in range(starting_col, ending_col):
            window = noisy_image[i - pad_size:i + pad_size + 1, j-pad_size:j+pad_size+1]
            window_mean = np.median(window)
            filtered_image[i][j] = window_mean

    return filtered_image 

average_filtered = average_filter(noise_image, 5)

median_filtered = median_filter(noise_image, 5)

psnr_noise = peak_signal_noise_ratio(image, noise_image)
psnr_average = peak_signal_noise_ratio(image, average_filtered)
psnr_median = peak_signal_noise_ratio(image, median_filtered)

plt.figure(figsize=(12, 6))


plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')


plt.subplot(2, 2, 2)
plt.imshow(noise_image, cmap='gray')
plt.title(f'Noisy Image\n PSNR : {psnr_noise:.2f} dB')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(average_filtered, cmap='gray')
plt.title(f'Average Filtered\nPSNR: {psnr_average:.2f} dB')
plt.axis('off')


plt.subplot(2, 2, 4)
plt.imshow(median_filtered, cmap='gray')
plt.title(f'Median Filtered\nPSNR: {psnr_median:.2f} dB')
plt.axis('off')

plt.tight_layout()
plt.show()
