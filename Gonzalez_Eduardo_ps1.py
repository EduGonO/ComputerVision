import numpy as np
from numpy import matlib
from scipy import misc
import matplotlib.pyplot as plt
import cv2

# Homework 1
# @Eduardo Gonzalez

m = np.array([[1,2, 3],[4, 5, 6],[7, 8, 9]])

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

print("m: ")
print(m, "\n")

print("a:", a, "\n")
print("b:", b, "\n")
print("c:", c, "\n")


# Dot product of a and b
aDotb = np.dot(a, b)
print("Dot Product of a and b:", aDotb, "\n")


# Element Wise Product of a and b
print("Element Wise Product of a and b:", np.multiply(a, b), "\n")


# Multiply each row of M by a (no for loop)
o = np.matlib.repmat(a, 3, 1) # Creates a 3x1 of [1, 2, 3]

new = np.multiply(o, m)

print("m * a:", "\n", new, "\n")


print(np.sort(new))

#-------------------------------------------------------------------

#     Part 2

#-------------------------------------------------------------------


print("OpenCV\n")

# Found that double presicion = np.float64()
image1 = np.float64(cv2.imread('/Users/edu/github/ComputerVision/im/image1.jpg'))
image2 = np.float64(cv2.imread('/Users/edu/github/ComputerVision/im/image2.jpg'))

cv2.imshow("image 1",image1) # Display image
cv2.imshow("image 2",image2) # Display image

im1 = cv2.normalize(image1, np.zeros((500, 500)), 0, 1, cv2.NORM_MINMAX)
im2 = cv2.normalize(image2, np.zeros((500, 500)), 0, 1, cv2.NORM_MINMAX)


# Cut the images in halves
i1 = im1[:, :250]
i2 = im2[:, 250:]

halves = np.concatenate((i1, i2), axis=1)

cv2.imwrite('halves.jpg', halves)

new = np.empty((500, 500))
for row in range(len(im1)):
    if row % 2 == 1: # es par
        new[row] = im1[row]
    else: # Se mueve de 1 y ya no es par
        new[row] = im2[row]

cv2.imwrite(new)

grey = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
