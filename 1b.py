import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'image2.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
num_bits = 1
quantization_step = 255 / (2 ** num_bits - 1)
quantized_images = []
while num_bits <= 8:
    quantized_image = (image / quantization_step).astype(np.uint8) * quantization_step
    quantized_images.append(quantized_image)
    
    num_bits += 1
    quantization_step = 255 / (2 ** num_bits - 1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('Quantized Images', fontsize=16)

for i, ax in enumerate(axes.flat):
    ax.imshow(quantized_images[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'{i+1} bits')
    ax.axis('off')

plt.tight_layout()
plt.show()