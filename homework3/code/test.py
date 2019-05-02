import numpy
# Only to read the image
from scipy import misc
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math

# Applies the gaussian filter with an already programed way of easyly doing a hig/lowPass
def gaussian_blur_kernel_2d(row, col, sigma, high):

    if row % 2 == 1:
        vCenter = int(row/2) + 1
    else:
        hCenter = int(row/2)

    if col % 2 == 1:
        hCenter = int(col/2) + 1
    else:
        hCenter = int(col/2)

    def gaussian(v,h):

        # ( -1 * (v-center^2 + h-center^2) ) / 2*sig^2
        coeff = math.exp(-1 * ((v - vCenter)**2 + (h - hCenter)**2) / (2 * sigma**2))

        if high:
            return (1-coeff)
        else:
            return coeff

    x = numpy.array([[gaussian(v,h) for h in range(col)] for v in range(row)])

    return x

# We use this to filter the matrix
def dft(matrix, matrix2):
   nDft = fftshift(fft2(matrix)) * matrix2
   return ifft2(ifftshift(nDft))

# This function creates a Low Pass Image
def lowPass(matrix, sigma):
   a, b = matrix.shape
   return dft(matrix, gaussian_blur_kernel_2d(a, b, sigma, high=False))

# This function creates a High Pass Image
def highPass(matrix, sigma):
   a, b = matrix.shape
   return dft(matrix, gaussian_blur_kernel_2d(a, b, sigma, high=True))


def hybridImage(highImgage, lowImgage, sigmaHigh, sigmaLow):
   high = highPass(highImgage, sigmaHigh)
   low = lowPass(lowImgage, sigmaLow)
   return high + low

p1 = misc.imread("/Users/edu/github/ComputerVision/homework3/data/cat.bmp", flatten=True)
p2 = misc.imread("/Users/edu/github/ComputerVision/homework3/data/dog.bmp", flatten=True)

hybrid = numpy.real(hybridImage(p1, p2, 25, 10))

misc.imsave("hybridImage.png", hybrid)
