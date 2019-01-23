import numpy as np
x = np.array([1,2,3,4])
w = np.array([1,-1,2])

res = np.convolve(x,w,'valid')
print(res)
