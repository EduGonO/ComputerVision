import numpy as np
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

print("Element Wise Product of a and b:", np.multiply(a, b), "\n")

# e = (m.transpose(0,1,3,2) * a).transpose(0,1,3,2)
e = a*m;
print("m*a:")
print(e,"\n")

print(np.sort(e))

## Part 2

image = cv2.imread('/Users/edu/github/ComputerVision/im/8f.jpg')
cv2.imshow('image', image)

# Convert to double presicion
img = image.astype(float)
print(img)

b = (img - np.min(img))/np.ptp(img)
print(b)
