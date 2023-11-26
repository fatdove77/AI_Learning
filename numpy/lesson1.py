import numpy as np
arr1 = np.arange(10)
cut = arr1[:3]
print(cut)
cut[0] = 100
print(arr1)
