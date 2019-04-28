import cv2
import numpy as np
import scipy
from scipy import misc
from scipy import ndimage
from scipy.ndimage import filters
import matplotlib.pyplot as plt


# Homework 2 - Eduardo Gonzalez


# Problem 1

# We read image 1
image1 = cv2.imread('/Users/edu/github/ComputerVision/Homework2/im/peppers.png')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image1 = np.float64(image1)

# We read image 2
image2 = cv2.imread('/Users/edu/github/ComputerVision/Homework2/im/cheetah.png')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image2 = np.float64(image2)

# We blur the images
image1blurred = ndimage.gaussian_filter(image1, 7)
image2blurred = ndimage.gaussian_filter(image2, 7)

# We display all the images
plt.figure(1, figsize=(15, 5))
plt.suptitle('Problem 1', fontsize=18)

plt.subplot(1, 4, 1)
plt.title('Image 1')
plt.imshow(image1, cmap=plt.cm.gray)

plt.subplot(1, 4, 2)
plt.title('Image 1 (Blurred)')
plt.imshow(image1blurred, cmap=plt.cm.gray)

plt.subplot(1, 4, 3)
plt.title('Image 2')
plt.imshow(image2, cmap=plt.cm.gray)

plt.subplot(1, 4, 4)
plt.title('Image 2 (Blurred)')
plt.imshow(image2blurred, cmap=plt.cm.gray)

plt.show()

# We compute the dft of the images

# For image 1
image1dft = np.fft.fft2(image1)
i1shift = np.fft.fftshift(image1dft)
image1dftFinal = np.log(np.abs(i1shift))

# For image 2
image2dft = np.fft.fft2(image2)
i1shift = np.fft.fftshift(image2dft)
image2dftFinal = np.log(np.abs(i1shift))

plt.figure(1, figsize=(15, 5))
plt.suptitle('Problem 1 (dft)')

plt.subplot(1, 2, 1)
plt.title('Image 1 (dft)')
plt.imshow(image1dftFinal, cmap=plt.cm.gray)

plt.subplot(1, 2, 2)
plt.title('Image 1 (dft)')
plt.imshow(image2dftFinal, cmap=plt.cm.gray)

plt.show()


# Problem 2


imageP2 = np.float64(misc.imread('/Users/edu/github/ComputerVision/Homework2/im/lowcontrast.jpg', flatten=1, mode = 'F'))


# Problem 3


einstein = np.float64(misc.imread('/Users/edu/github/ComputerVision/Homework2/im/einstein.png', flatten=1, mode = 'F'))

# We store the Gaussian Filter
gaussian = 1.0/256*np.array([[ 1,  4,  6,  4, 1],
                                [ 4, 16, 24, 16, 4],
                                [ 6, 24, 36, 24, 6],
                                [ 4, 16, 24, 16, 4],
                                [ 1,  4,  6,  4, 1]])

gaussianX = np.array([1,4,6,4,1])
gaussianY = 1.0/256*np.array([1,4,6,4,1])

# We store the Sobel Filter
sobel = np.array([[ 1,  2, 0,  -2, -1],
                     [ 4,  8, 0,  -8, -4],
                     [ 6, 12, 0, -12, -6],
                     [ 4,  8, 0,  -8, -4],
                     [ 1,  2, 0,  -2, -1]])

sobelX = np.array([1,2,0,-2,-1])
sobelX = np.array([1,4,6,4,1])

# We store the Box Filter
box = np.ones((5,5), dtype=np.float32)/25
boxX = np.array([1,1,1,1,1])
boxY = np.array([1,1,1,1,1])/25.0

# We apply the Gaussian, Box and Sobel filters to the images
einsteinGaussian = filters.convolve(einstein, gaussian, mode="mirror")
einsteinBox = filters.convolve(einstein, box, mode="wrap")
einsteinSobel = filters.convolve(einstein, sobel, mode="nearest")

# We apply the Gaussian, Box and Sobel filters to the images in their X axis
einsteinGaussianX = filters.convolve1d(einstein, gaussianX, axis=1, mode = "mirror")
einsteinBoxX = filters.convolve1d(einstein, boxX, axis=1, mode="wrap")
einsteinSobelX = filters.convolve1d(einstein, sobelX, axis=1, mode="nearest")

# We apply the Gaussian, Box and Sobel filters to the images in their Y axsis
einsteinGaussianY = filters.convolve1d(einstein, gaussianY, axis=0, mode = "mirror")
einsteinBoxY = filters.convolve1d(einstein, boxY, axis=0, mode="wrap")
einsteinSobelX = filters.convolve1d(einstein, sobelX, axis=0, mode="nearest")

# We show the Gaussian images
finalGaussian = plt.figure()
finalGaussian.suptitle("Gaussian Convolution")

einsteinPlot = finalGaussian.add_subplot(2,2,1)
einsteinPlot.set_title("Einstein")
einsteinPlot.imshow(einstein, cmap=plt.cm.gray)

einsteinGaussianPlot = finalGaussian.add_subplot(2,2,2)
einsteinGaussianPlot.set_title("Gaussian")
einsteinGaussianPlot.imshow(einsteinGaussian, cmap=plt.cm.gray)

einsteinGaussianXPlot = finalGaussian.add_subplot(2,2,3)
einsteinGaussianXPlot.set_title("Gaussian (X Axis)")
einsteinGaussianXPlot.imshow(einsteinGaussianX, cmap=plt.cm.gray)

einsteinGaussianYPlot = finalGaussian.add_subplot(2,2,4)
einsteinGaussianYPlot.set_title("Gaussian (Y Axis)")
einsteinGaussianYPlot.imshow(einsteinGaussianY, cmap=plt.cm.gray)

# We show the Box images
finalBox = plt.figure()
finalBox.suptitle("Box Convolution")

einsteinPlot = finalBox.add_subplot(2,2,1)
einsteinPlot.set_title("Einstein")
einsteinPlot.imshow(einstein, cmap=plt.cm.gray)

einsteinBoxPlot = finalBox.add_subplot(2,2,2)
einsteinBoxPlot.set_title("Box")
einsteinBoxPlot.imshow(einsteinBox, cmap=plt.cm.gray)

einsteinBoxXPlot = finalBox.add_subplot(2,2,3)
einsteinBoxXPlot.set_title("Box (Y Axis)")
einsteinBoxXPlot.imshow(einsteinBoxX, cmap=plt.cm.gray)

einsteinBoxYPlot = finalBox.add_subplot(2,2,4)
einsteinBoxYPlot.set_title("Box (Y Axis)")
einsteinBoxYPlot.imshow(einsteinBoxY, cmap=plt.cm.gray)

# We show the Sobel images
finalSobel = plt.figure()
finalSobel.suptitle("Sobel Convolution")

einsteinPlot = finalSobel.add_subplot(2,2,1)
einsteinPlot.set_title("Einstein")
einsteinPlot.imshow(einstein, cmap=plt.cm.gray)

einsteinSobelPlot = finalSobel.add_subplot(2,2,2)
einsteinSobelPlot.set_title("Sobel")
einsteinSobelPlot.imshow(einsteinSobel, cmap=plt.cm.gray)

einsteinSobelXPlot = finalSobel.add_subplot(2,2,3)
einsteinSobelXPlot.set_title("Sobel (X Axis)")
einsteinSobelXPlot.imshow(einsteinSobelX, cmap=plt.cm.gray)

einsteinSobelXPlot = finalSobel.add_subplot(2,2,4)
einsteinSobelXPlot.set_title("Sobel (Y Axis)")
einsteinSobelXPlot.imshow(einsteinSobelX, cmap=plt.cm.gray)

plt.show()


# Problem 4 


zebra = np.float64(misc.imread('/Users/edu/github/ComputerVision/Homework2/im/zebra.png', flatten=1, mode='F'))

# We aply on different axes
zebraX = ndimage.sobel(zebra, 0)
zebraY = ndimage.sobel(zebra, 1)

#magnitude
m = np.hypot(zebraX, zebraY)

plt.figure(figsize=(10, 5))
plt.suptitle('Problem 4', fontsize=18)

plt.subplot(1, 4, 1)
plt.imshow(zebra, cmap=plt.cm.gray)
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(zebraX, cmap=plt.cm.gray)
plt.title('Edges (X Axis)')

plt.subplot(1, 4, 3)
plt.imshow(zebraY, cmap=plt.cm.gray)
plt.title('Edges (Y Axis)')

plt.subplot(1, 4, 4)
plt.imshow(m, cmap=plt.cm.gray)
plt.title('Edges (X & Y Axis)')

plt.show()