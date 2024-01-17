import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("stone.jpg", cv2.IMREAD_GRAYSCALE)

[rows, cols] = image.shape

f = 2
sampled_image = []
for k in range(8):
    temp_image = np.zeros((rows//f, cols // f), dtype=np.uint8)

    if temp_image.shape[0] > 0 and temp_image.shape[1] > 0:
        for i in range(0, rows, f):
            for j in range(0, cols, f):
                temp_image[i // f][j // f] = image[i][j]
        sampled_image.append(temp_image)
    f = f*2


row, col = 2, 4
plt.figure(figsize=(14,9))
idx = 0
for i in range(row):
    for j in range(col):
        plt.subplot(row, col, idx+1)
        plt.imshow(sampled_image[idx], cmap='gray')
        [h,w] = sampled_image[idx].shape
        plt.title(f'{h}x{w}')
        idx += 1
plt.show()