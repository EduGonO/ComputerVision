import numpy as np
import matplotlib.pyplot as plt
import cv2

# Numpy warmup
# @Eduardo Gonzalez

print("")
print("Numpy Warmup")
print("By: Eduardo Gonzalez")
print("")

# 1.  3x3 Identity Matrix
print("1.")

x = np.eye(3)
print(x, "\n")



# 2.  3x3 Array of Random Values
print("2.")

x = np.random.randint(10, size=(3, 3))
print(x, "\n")



# 3.  10x10 Array of Random Values + min and max
print("3.")

x = np.random.randint(10, size=(9, 9))
print(x, "\n")

print("Min: ", x.min())
print("Max: ", x.max(), "\n")



# 4.  Add a Border of 0 to an Existing Matrix
print("4.")

x = np.ones((3, 3))
print(x, "\n")

x = np.pad(x, pad_width=1, mode='constant', constant_values=0)
print(x, "\n")



# 5.  Random Vector of size 40 + mean
print("5.")

x = np.random.randint(0, 101, 40)
print(x, "\n")
print("Mean: ", np.mean(x), "\n")



# 6.  Checkerboard 8x8 using tile function
print("6.")

x = np.array([[0,1],[1,0]])
print(np.tile(x,(4, 4)), "\n")



# 7.  Vector of 100 Uniformy Distributed Values from 0 to 1
print("7.")

x = np.random.uniform(0,1,100)
print(x, "\n")

#------------------------------------------------------------

# Matplotlib

# Vector of 1000 random values drawn from a normal distribution
# with a mean of 0 and standard derivation of 0.5
print("Matplotlib\n")

std = 0.5 #Standard Deviation

x = np.random.normal(0, std, 1000)

print("STD: ", np.std(x))
print("Mean: ", np.mean(x), "\n")

plt.plot(x)
plt.axis([0, 1000, 0, 2]) # Exclude negative numbers on the plot
plt.show()

#------------------------------------------------------------

# OpenCV

# Import an image, covert to grayscale, export it as .png and find the
# brightest and darkest pixel values of the image
print("OpenCV\n")

image = cv2.imread('/Users/edu/github/ComputerVision/im/8f.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
#cv2.imwrite('gray.png',gray)
imageGrey = cv2.resize(gray, (1000, 823)) # Resize since the image is too big

cv2.imshow('gray',imageGrey) # Display image

print("Brightest Pixel: ", imageGrey.max())
print("Darkest Pixel: ", imageGrey.min(), "\n")

cv2.waitKey(0) # Waits for a key to be pressed
cv2.destroyAllWindows() # Closes all windows
