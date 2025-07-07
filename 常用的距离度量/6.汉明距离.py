import numpy as np
a=np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0])
b=np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
hanm_dis = np.count_nonzero(a!=b)
print("汉明距离为", hanm_dis)