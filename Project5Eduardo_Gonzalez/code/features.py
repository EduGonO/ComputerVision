import cv2

from scipy import signal
from scipy import misc
from scipy.ndimage import gaussian_filter

from pylab import *

# Project 5: Eduardo Gonzalez

# Outputs a standard Gaussian Kernel
def gaussianKernel(size):

    x, y = mgrid[-size:size+1, -size:size+1]
    g = exp(-(x**2/float(size)+y**2/float(size)))
    return g / g.sum()

def imageCorners(image):

    size = 3

    y, x = mgrid[-size:size+1, -size:size+1]

    # We create a gaussianX and a gaussianY
    gaussianX = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*size)**2)))
    gaussianY = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*size)**2)))

    imageX = signal.convolve(im, gaussianX, mode='same')
    imageY = signal.convolve(im, gaussianY, mode='same')

    # Gaussian filter to blur the image
    gaussian = gaussianKernel(3)

    # We combine the possible outcomes
    finalXX = signal.convolve(imageX*imageX, gaussian, mode='same')
    finalXY = signal.convolve(imageX*imageY, gaussian, mode='same')
    finalYY = signal.convolve(imageY*imageY, gaussian, mode='same')

    aux = finalXX * finalYY - finalXY**2

    return aux / (finalXX + finalYY)

def getPoints(im):

    # We find the coordanates for the top corner above 0.1 (Threshold)
    # Modify the Threshold to find more/less features
    corner = (im > max(im.ravel()) * 0.2).nonzero()
    arrayCoordinates = [(corner[0][k], corner[1][k]) for k in range(len(corner[0]))]

    # We fill an array with all the values we found
    val = argsort([im[j[0]][j[1]] for j in arrayCoordinates])

    # We create an array to store all, this also contains the min number of pixels
    # between points (I decided 10 pixels)
    pointsArray = zeros(im.shape)
    pointsArray[10:-10, 10:-10] = 1

    final = []

    for i in val:
        if pointsArray[arrayCoordinates[i][0]][arrayCoordinates[i][1]] == 1:
            final.append(arrayCoordinates[i]) # if == 1, append
            pointsArray[(arrayCoordinates[i][0]-10):(arrayCoordinates[i][0]+10), (arrayCoordinates[i][1]-10):(arrayCoordinates[i][1]+10)] = 0

    return final







# Read the image
im = misc.imread('../data/Notre Dame/4191453057_c86028ce1f_o.jpg',flatten=1)
imDisplay = misc.imread('../data/Notre Dame/4191453057_c86028ce1f_o.jpg')

#im = misc.imread('../data/Notre Dame/921919841_a30df938f2_o.jpg',flatten=1)
#imDisplay = misc.imread('../data/Notre Dame/921919841_a30df938f2_o.jpg')

# Get the points from the data that we got from the corners
points = getPoints(imageCorners(im))

# Plot
plt.gray()
plt.imshow(imDisplay)
plot([p[1] for p in points],[p[0] for p in points],'o')
plt.axis('off')
plt.show()
