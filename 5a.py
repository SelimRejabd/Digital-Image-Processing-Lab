import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('dilation.png', cv2.IMREAD_GRAYSCALE)

def perform_erosion(image, structuring_element_size):
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)
    rows, cols = image.shape
    eroded_image = np.zeros_like(image)
    pad_size = structuring_element_size // 2
    for i in range(pad_size, rows - pad_size):
        for j in range(pad_size, cols - pad_size):
            region = image [i-pad_size : i+ pad_size,
                            j - pad_size : j + pad_size]
            eroded_image[i][j] = np.min(region)
    return eroded_image

def dilation(image, struturing_element_size):
    kernel = np.ones((struturing_element_size, struturing_element_size), np.uint8)
    rows, cols = image.shape
    dilated_image = np.zeros_like(image)
    pad_size = struturing_element_size // 2

    for i in range(pad_size, rows - pad_size):
        for j in range(pad_size, cols - pad_size):
            region = image[i -pad_size : i+pad_size,
                           j - pad_size: j+pad_size]
            dilated_image[i][j] = np.max(region)
    return dilated_image

dilated_image = dilation(image, 5)
eroded_image = perform_erosion(image, 5)
plt.figure(figsize=(12,7))
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original image")

plt.subplot(1,3,2)
plt.imshow(eroded_image, cmap='gray')
plt.title("Eroded image")
plt.subplot(1,3,3)
plt.imshow(dilated_image, cmap='gray')
plt.title("Dilated image")
plt.show()