import numpy as np
from matplotlib import pyplot as plt

# # 一、向量初始化
# a = np.array([1.,2,3,4,5])
# print(a)
# print(a.dtype)
# print(a.ndim)
#
# b = np.zeros(5, dtype=float)
# print(b)
#
# c = np.zeros_like(a)
# print(c)
# print(c.dtype)
#
# d = np.ones(4)
# print(d)
#
#
# e = np.empty(4)
# print(e)
#
# f = np.full(4, 5.555)
# print(f)
#
# arr1 = np.arange(10)
# arr2 = np.arange(3, 10)
# arr3 = np.arange(3, 10,2)
# print(arr1)
# print(arr2)
# print(arr3)
#
#
# print(np.arange(0.4, 0.8, 0.1))
# print(np.arange(0.5, 0.8, 0.1))
# print(np.linspace(0, 1, 5))
#
# data = np.random.randint(1, 10, 3000)
# data = np.random.randn(3000)
# data = np.random.normal(5, 20, 3000)
# data = np.random.uniform(0, 1, 3000)
# data = np.random.rand(3000)


# # 设置图片清晰度
# plt.rcParams['figure.dpi'] = 300
#
# # 绘制直方图
# plt.hist(data, bins=300, edgecolor='black')
# plt.title('Standard Normal Distribution Histogram')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.grid(True)
#
# # 显示图形
# plt.show()


# a = np.arange(1, 6)
# print(a[1])
# print(a[2:4])
# print(a[-2:])
# print(a[::2])
# print(a[[1,3,4]])
#
# # a[2:4] = 0
# a[[1,3,4]] = 100
# print(a)
#
# print('-----------------')
#
# a = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])
# print(a[a > 5])
#
# print('-----------------')
# print(np.where(a > 5))
#
# # 三参数形式
# a = np.array([1, 2, 3, 4, 5])
# b = np.array([10, 20, 30, 40, 50])
# condition = np.array([True, False, True, False, True])
# result = np.where(condition, a, b)
# print("三参数形式结果:", result)
#
#
# arr = np.array([-1, 2, 5, 8, 10])
# clipped_arr = np.clip(arr, 1, 7)
# print("np.clip 结果:", clipped_arr)


# a = np.array([1,2])
# b = np.array([3,4])
# print(a + b)
# print(a - b)
# print(a * b)
# print(a / b)
# print(a % b)
# print(a ** b)
# print(a // b)

# a = np.array([1,2,3,4,5])
# print(a + 10)
# print(a - 10)
# print(a * 10)
# print(a / 10)

# print(np.array([1,2,3,4]) ** 2)
# print(np.sqrt([1,4,9,16]))
# print(np.sqrt(np.array([1,4])))
# print(np.exp([1,2,3,4]))
# print(np.log([np.e, np.e**2]))


# print(np.dot([1,1,1], [3,3,3]))
# print(np.cross([1,1,1], [3,3,3]))
#
#
# print(np.sin([np.pi/2, np.pi/4, np.pi/6]))
#
#
# print(np.floor([1.2, 1.5, 1.8, 2.1]))
# print(np.ceil([1.2, 1.5, 1.8, 2.1]))
# print(np.round([1.2, 1.5, 1.8, 2.1], 1))


print(np.max([1,2,3,4,5]))
print(np.min([1,2,3,4,5]))
print(np.mean([1,2,3,4,5]))
print(np.median([1,2,3,4,5]))
print(np.std([1,2,3,4,5]))
print(np.var([1,2,3,4,5]))
print(np.sum([1,2,3,4,5]))
print(np.prod([1,2,3,4,5]))
print(np.argmax([1,2,3,4,5]))
print(np.argmin([1,2,3,4,5]))
print(np.argsort([1,2,3,4,5]))















