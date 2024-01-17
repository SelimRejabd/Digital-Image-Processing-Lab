import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('stone.jpg', cv2.IMREAD_GRAYSCALE)
# image=cv2.resize(image,(512,512))
[m, n] = image.shape

f = 2
sampled_image = []
for k in range(8):
    img2 = np.zeros((m // f, n // f), dtype=np.uint8)

    if img2.shape[0] > 0 and img2.shape[1] > 0:
        for i in range(0, m, f):
            for j in range(0, n, f):
                    img2[i // f][j // f] = image[i][j]
        sampled_image.append(img2)
    f = f*2


row, col = 2, 4
plt.figure(figsize=(14,9))
idx = 0
for i in range(row):
    for j in range(col):
        plt.subplot(row,col, idx+1)
        plt.imshow(sampled_image[idx], cmap='gray')
        [h,w] = sampled_image[idx].shape
        plt.title(f'{h}x{w}')
        idx+=1
# plt.tight_layout()
plt.show()