import numpy as np
import matplotlib.pyplot as plt

# Numpy warmup
# @Eduardo Gonzalez


m = np.matrix('1, 2; 3, 4')
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

print("a: ")
print(a, "\n")
print("b: ")
print(b, "\n")
print("c: ")
print(c, "\n")

# Dot product of a and b
print("Dot Product of a and b: ", np.dot(a, b))

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