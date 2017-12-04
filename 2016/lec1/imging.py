import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img= imread('cat1.jpg')
print(img.shape)
img_tinted = img* [1,0.95,0.9]
print(img.shape)
plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(np.uint8(img_tinted))
print(img[0][0][0])