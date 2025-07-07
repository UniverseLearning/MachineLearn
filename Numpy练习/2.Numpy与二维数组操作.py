import numpy as np

# a = np.array([[1,2,3],[4,5,6]])
# print(a)
# print(a.shape)
# print(a.ndim)

# print(np.zeros((2,3)))
# print(np.ones((2,3)))
# print(np.full((2,3), 9))
# print(np.zeros((2,3)))
# print(np.eye(3,1))

# print(np.zeros([2,33]))
#
# data1 = np.random.randint(low=0, high=5, size=(5, 5))
# data2 = np.random.rand(3,2)
# data3 = np.random.uniform(1, 10,[3,2])
# data4 = np.random.randn(3,2)
# data5 = np.random.normal(10, 2, [3,2,4])

# print(data1)
# print(data2)
# print(data3)
# print(data4)
# print(data5)


# data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# print(data[1,2])
# print(data[1,:])
# print(data[:,2])
# print(data[:,1:3])
# print(data[-2:,-2:])
# print(data[::2,1::2])

# data = np.array([[1,2,3],[4,5,6]])
# print(data.sum())
# print(data.sum(axis=0))
# print(data.sum(axis=1))


# a = np.array([[1,2],[3,4]])
# b = np.array([[1,0],[0,1]])
# print("a + b: \n", a + b)
# print("a - b: \n", a - b)
# print("a * b: \n", a * b)
# print("a @ b: \n", a @ b)
# print("a ** b: \n", a ** b)


# data = np.array([[1,2,3],[4,5,6],[7,8,9]])
# a = np.array(9)
# print(data / a)
# a = np.array([-1,0,1])
# print(data * a)
# a = np.array([[3],[6],[9]])
# print(data / a)


# a = np.array([[1],[2],[3]])
# b = np.array([[1,2,3]])
# print(a)
# print(b)
# print(a.ndim, a.shape)
# print(b.ndim, b.shape)
# print(a @ b)
# print(b @ a)


# print("------------------")
# print(np.array([[1,2,3],[4,5,6]]).T)
# print(np.array([1,2,3]).T)
# print(np.array([[1,2,3]]).T)

# data = np.array([1,2,3,4,5,6])
# print(data.reshape(2,-1))

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b = np.array([[1,2,3,4],[5,6,7,8]])
c = np.array([[1,2],[3,4],[5,6]])
print(np.hstack((a,c)))
print(np.vstack((a,b)))





























