import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1
im = ndimage.rotate(im, 15, mode='constant')

print("im: ", "\n", im)
# 2. Blur the image using a Gaussian filter
im = ndimage.gaussian_filter(im, 8)

# 3. Apply Sobel filter to both x and y direction.
sx =  ndimage.sobel(im, axis=0, mode='constant')

print("sx: ", "\n", sx)
