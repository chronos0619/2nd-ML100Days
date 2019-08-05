
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

#%matplotlib inline
#內嵌繪圖並省去plt.show()

import numpy as np
import cv2
import skimage.io as skio
import timeit

img1 = skio.imread('data/呱吉_1.PNG')
plt.imshow(img1)
plt.show()

img2 = Image.open('data/習維尼.png') #PIL object
img2 = np.array(img2)
plt.imshow(img2)
plt.show()
#要關掉第一張圖才會顯示第二張圖

"""
img3 = cv2.imread('data/巨槌瑞斯.jpg')
plt.imshow(img3)
plt.show()

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
plt.imshow(img3)
plt.show()
"""

N_times = 1000

#%%timeit
im = np.array([skio.imread('data/巨槌瑞斯.jpg') for _ in range(N_times)])
