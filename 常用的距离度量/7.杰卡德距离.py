import numpy as np

vec1 = np.random.random(10) > 0.5
vec2 = np.random.random(10) > 0.5
vec1 = np.asarray(vec1, np.int32)
vec2 = np.asarray(vec2, np.int32)
up = np.double(np.bitwise_and((vec1 != vec2), np.bitwise_or(vec1 != 0, vec2 != 0)).sum())
down = np.double(np.bitwise_or(vec1 != 0, vec2 != 0).sum())
jaccard_dis = 1 - (up / down)
print("杰卡德距离为", jaccard_dis)
