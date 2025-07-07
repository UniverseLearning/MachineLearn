import numpy as np
vector1 = np.array([1,2,3])
vector2 = np.array([4,5,6])
cos_sim = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
print("余弦相似度为", cos_sim)