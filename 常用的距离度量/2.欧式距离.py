import numpy as np

vector1 = np.array([1,2,3])
vector2 = np.array([4,5,6])

eud_distance = np.sqrt(np.sum((vector1 - vector2) ** 2))
print(eud_distance)

print(type(eud_distance))

print(3 * np.sqrt(3))