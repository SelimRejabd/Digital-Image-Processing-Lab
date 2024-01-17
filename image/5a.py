import numpy as np
import matplotlib.pyplot as plt
import cv2
def custom_erosion(image, kernel):
    result = np.zeros_like(image)
    rows, cols = image.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if np.all(image[i-1:i+2, j-1:j+2] * kernel):
                result[i, j] = 0

    return result

def custom_dilation(image, kernel):
    result = np.zeros_like(image)
    rows, cols = image.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if np.any(image[i-1:i+2, j-1:j+2] * kernel):
                result[i, j] = 1

    return result

# Example usage
image_path = 'fingerprint.png'
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is binary (black and white)
binary_image = (binary_image > 0.5).astype(np.uint8)

# Define the structuring element (kernel)
# kernel_size = 3
# kernel = np.ones((kernel_size, kernel_size), np.uint8)
kernel = [[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]

# Manually perform erosion
erosion_result = custom_erosion(binary_image, kernel)

# Manually perform dilation
dilation_result = custom_dilation(binary_image, kernel)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(binary_image, cmap='gray')
plt.title('Original Binary Image')

plt.subplot(132)
plt.imshow(erosion_result, cmap='gray')
plt.title('Manual Erosion')

plt.subplot(133)
plt.imshow(dilation_result, cmap='gray')
plt.title('Manual Dilation')

plt.show()
