# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:39:14 2015

@author: bxiao
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc

image = misc.imread('/Users/edu/github/ComputerVision/homework3/data/einstein.bmp', flatten=1)
image = np.float64(image)

#[intput_row, intput_col] = size(image(mslice[:], mslice[:], 1))
#[filter_row, filter_col] = size(filter)

# Pad image with zeros (amount = minimum need of filter = half of row and
# column

pad_input_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

plt.figure(1, figsize=(15, 5))
plt.suptitle('Problem', fontsize=18)

plt.subplot(1, 4, 1)
plt.title('Image with pad')
plt.imshow(pad_input_image, cmap=plt.cm.gray)

gaussian = 1.0/256*np.array([[ 1,  4,  6,  4, 1],
                             [ 4, 16, 24, 16, 4],
                             [ 6, 24, 36, 24, 6],
                             [ 4, 16, 24, 16, 4],
                             [ 1,  4,  6,  4, 1]])

print(gaussian)
