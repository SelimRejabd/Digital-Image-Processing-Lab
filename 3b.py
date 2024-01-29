import cv2
import numpy as np
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
print(type(noisy_image))
add_salt_noise(noisy_image)
add_papper_noise(noisy_image)


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
            window = noisy_image[i - pad_size:i +
                                 pad_size + 1, j-pad_size:j+pad_size+1]
            temp_window = window.copy()
            temp_window = temp_window * mask
            temp_window_mean = np.sum(temp_window)
            filtered_image[i][j] = temp_window_mean

    return filtered_image


def calculate_psnr(image, filtered_image):
    image = image.astype(np.float64)
    filtered_image = filtered_image.astype(np.float64)
    mse = np.mean((image-filtered_image)**2)
    max_pixel = 255.0
    psnr = 20*np.log10(max_pixel/np.sqrt(mse))
    return psnr


mask_3_filtered_image = average_filter(noisy_image, 3)
psnr_for_mask_3 = calculate_psnr(noisy_image, mask_3_filtered_image)

mask_5_filtered_image = average_filter(noisy_image, 5)
psnr_for_mask_5 = calculate_psnr(noisy_image, mask_5_filtered_image)

mask_7_filtered_image = average_filter(noisy_image, 7)
psnr_for_mask_7 = calculate_psnr(noisy_image, mask_7_filtered_image)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mask_3_filtered_image, cmap='gray')
plt.title(f'Avarage filtered image (3x3)  PSNR {psnr_for_mask_3:.2f} dB')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mask_5_filtered_image, cmap='gray')
plt.title(f'Avarage filtered image (5x5)  PSNR {psnr_for_mask_5:.2f} dB')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(mask_7_filtered_image, cmap='gray')
plt.title(f'Avarage filtered image (7x7)  PSNR {psnr_for_mask_7:.2f} dB')
plt.axis('off')
plt.show()
