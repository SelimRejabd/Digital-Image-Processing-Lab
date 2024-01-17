import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


def perform_erosion(image, structuring_element_size):
    # Define the structuring element (kernel)
    kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)

    # Get image dimensions
    rows, cols = image.shape

    # Create an empty array for the result
    eroded_image = np.zeros_like(image)

    # Perform Erosion manually
    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2:i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2:j + structuring_element_size // 2 + 1]
            eroded_image[i, j] = np.min(region)

    return eroded_image


def perform_dilation(image, structuring_element_size):
    # Define the structuring element (kernel)
    # kernel = np.ones((structuring_element_size, structuring_element_size), np.uint8)
    karnel = [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]]
    # karnel = [[1, 1, 1],
    #           [1, 1, 1],
    #           [1, 1, 1]]  
    # Get image dimensions
    rows, cols = image.shape

    # Create an empty array for the result
    dilated_image = np.zeros_like(image)

    # Perform Dilation manually
    for i in range(structuring_element_size // 2, rows - structuring_element_size // 2):
        for j in range(structuring_element_size // 2, cols - structuring_element_size // 2):
            region = image[i - structuring_element_size // 2:i + structuring_element_size // 2 + 1,
                           j - structuring_element_size // 2:j + structuring_element_size // 2 + 1]
            dilated_image[i, j] = np.max(region)

    return dilated_image


# load an image
image1 = cv2.imread('./aa/opening.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./aa/dilation.png', cv2.IMREAD_GRAYSCALE)
structuring_element_size = 3  # Adjust the size as needed

# Perform Erosion
eroded_image = perform_erosion(image1, structuring_element_size)
# Perform Dilation
dilated_image = perform_dilation(image2, structuring_element_size)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
pl.title("Original Image1")
plt.imshow(image1, cmap='gray')

plt.subplot(2, 2, 2)
pl.title("Erosion Operation")
plt.imshow(eroded_image, cmap='gray')

plt.subplot(2, 2, 3)
pl.title("Orignal Image2")
plt.imshow(image2, cmap='gray')

plt.subplot(2, 2, 4)
pl.title("Dilation Operation")
plt.imshow(dilated_image, cmap='gray')

plt.tight_layout()
plt.show()