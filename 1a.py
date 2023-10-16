import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('stone.jpg', cv2.IMREAD_GRAYSCALE)
img1=cv2.resize(img1,(512,512))
[m, n] = img1.shape
print(m,n)

f = 2
sampled_image = []
for k in range(8):
    img2 = np.zeros((m // f, n // f), dtype=np.uint8)

    if img2.shape[0] > 0 and img2.shape[1] > 0:
        for i in range(0, m, f):
            for j in range(0, n, f):
                    img2[i // f][j // f] = img1[i][j]
        sampled_image.append(img2)
    f = f*2


row, col = 2, 4
fig, ax = plt.subplots(row, col, figsize=(9, 7))
idx = 0
for i in range(row):
    for j in range(col):
        ax[i, j].imshow(sampled_image[idx], cmap='gray')
        h = sampled_image[idx].shape[0]
        w = sampled_image[idx].shape[1]
        ax[i, j].set_title(f'{h}x{w}')
        idx+=1

plt.show()