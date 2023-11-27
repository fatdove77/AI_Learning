import numpy as np
arr1 =np.random.normal(500,70,1000)
print(np.where(arr1>650)[0])
print(np.where(arr1==np.max(arr1)))