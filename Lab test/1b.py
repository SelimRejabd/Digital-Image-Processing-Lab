import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("stone.jpg", cv2.IMREAD_GRAYSCALE)

sampled_image = []
sampled_image.append(image.copy())
rows, cols = image.shape

for k in range(7):
    for i in range(rows):
        for j in range(cols):
            image[i][j] = image[i][j] >> 1
    sampled_image.append(image.copy())

row, col = 2, 4
plt.figure(figsize=(12,7))
idx = 0
for i in range(row):
    for i in range(col):
        plt.subplot(row, col, idx+1)
        plt.imshow(sampled_image[idx], cmap='gray')
        plt.title(f'{8-idx}bits')
        idx += 1
plt.show()