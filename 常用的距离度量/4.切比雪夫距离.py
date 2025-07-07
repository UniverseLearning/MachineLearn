import numpy as np


vector1 = np.array([1, 2, 3])
vector2 = np.array([5, 5, 6])

qiebixuefu_distance = np.max(np.abs(vector1 - vector2))
print(qiebixuefu_distance)