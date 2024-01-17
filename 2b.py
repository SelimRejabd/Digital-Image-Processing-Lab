import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sun.jpeg', cv2.IMREAD_GRAYSCALE)

gamma = 0.5

c = 1.0

power_law_transformed = c * np.power(image, gamma)

# inverse_logarithm_image = c * np.exp(image/255)
# inverted_image = np.clip(inverse_logarithm_image, 0, 255)
# inverse_logarithm_image = np.uint8(inverse_logarithm_image)
# logarithm_image = c * np.log(1+image)

def inverse_log(image):
    # Perform the inverse log operation using np.exp
    inverted_image = np.exp(image/255)
    
    # If the image data type is float, you may want to normalize it back to the valid pixel range (e.g., 0 to 255)
    if image.dtype == np.float64:
        inverted_image = (inverted_image / np.max(inverted_image)) * 255
    
    # If the image data type is integer, clip the values to the valid pixel range
    else:
        inverted_image = np.clip(inverted_image, 0, 255)
    
    # Convert the image back to the appropriate data type
    inverted_image = inverted_image.astype(image.dtype)
    
    return inverted_image
inverse_log_image = inverse_log(image)

def inverse_log(image):
    inverse_log_image = image.copy()
    inverse_log_image = inverse_log_image/255.0
    r = np.arange(0, 256)
    c = 255.0 / np.log(1 + 255)
    # c = 1
    # for inverse log operation
    y = (np.exp(r) ** (1/c)) - 1
    [height, width] = inverse_log_image.shape
    for i in range(height):
        for j in range(width):
            inverse_log_image[i][j] = np.exp(inverse_log_image[i][j]*255) ** (1/c) - 1

    return inverse_log_image, y, r

log_image, log_f_value, log_f_range = inverse_log(image)

plt.figure(figsize=(12, 4))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(power_law_transformed, cmap='gray')
plt.title('Power-Law Transformed')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(inverse_log_image, cmap='gray')
plt.title('Inverse Logarithmic Transformed')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(log_image, cmap='gray')
plt.title('inverse log image')
plt.axis('off')


# plt.tight_layout()
plt.show()
