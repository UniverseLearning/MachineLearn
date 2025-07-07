import numpy as np


vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

manhaton_distance = np.sum(np.abs(vector1 - vector2))
print(manhaton_distance)