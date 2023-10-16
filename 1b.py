import cv2
import numpy as np
import matplotlib.pyplot as plt

gray_image = cv2.imread('stone.jpg', cv2.IMREAD_GRAYSCALE)

[height, width] = gray_image.shape
sampled_image = []
sampled_image.append(gray_image.copy())

for k in range(7):
    for i in range(height):
        for j in range(width):
            gray_image[i][j] = gray_image[i][j] >> 1
    sampled_image.append(gray_image.copy())
row, col = 2, 4
idx = 0
plt.figure(figsize=(12, 6))

for i in range(row):
    for j in range(col):
        plt.subplot(row, col, idx+1)
        plt.imshow(sampled_image[idx], cmap='gray')
        plt.title(f'{8-idx}bits')
        idx+=1

plt.show()