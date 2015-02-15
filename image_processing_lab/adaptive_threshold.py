'''
Created on Feb 5, 2015

@author: rostislavrypl
'''

import matplotlib.pyplot as plt

from skimage import data
from skimage.filter import threshold_otsu, threshold_adaptive
import matplotlib.image as mpimg
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

img = mpimg.imread('specimen.jpg')
# crop figure 
image = rgb2gray(img)[100:1000,200:1200]

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 50
binary_adaptive = threshold_adaptive(image, block_size, offset=2)

fig, axes = plt.subplots(ncols=3, figsize=(20, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()