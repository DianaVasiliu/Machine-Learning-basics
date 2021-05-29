import numpy as np


x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])


# ### Sum by elements

print(x + y)
print(np.add(x, y))


# ### Difference by elements

print(x - y)
print(np.subtract(x, y))


# ### Multiply by elements

print(x * y)
print(np.multiply(x, y))


# ### Division by elements

print(x / y)
print(np.divide(x, y))


# ### Square root by elements

print(np.sqrt(x))


# ### Power by elements

my_array = np.arange(5)
powered = np.power(my_array, 3)
print(powered)


# ### Multiplying with a scalar

v = np.array([9, 10])
w = np.array([11, 12])

# array * array
print(v.dot(w))
print(np.dot(v, w))

# matrix * array
print(np.matmul(x, v))

# matrix * matrix
print(np.matmul(x, y))


# ## Matrix operations
#
# ### Transpose & inverse

# matrix transpose
my_array = np.array([[1, 2, 3], [4, 5, 6]])
print(my_array.T)

# matrix inverse
my_array = np.array([[1., 2.], [3., 4.]])
print(np.linalg.inv(my_array))


# ### Sum on specific dimension

x = np.array([[1, 2], [3, 4]])

print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

# we can specify more axis
print(np.sum(x, axis=(0, 1)))


# ### Mean on specific dimension

y = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]],
              [[1, 2, 3, 4], [5, 6, 7, 8]],
              [[1, 2, 3, 4], [5, 6, 7, 8]]])
print(y.shape)
print(y)

print()

print(np.mean(y, axis=0))
print(np.mean(y, axis=1))


# ### The index of the maximum element on each line

z = np.array([[10, 12, 5], [17, 11, 19]])
print(np.argmax(z, axis=1))
