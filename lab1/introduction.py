import numpy as np


# ## MULTIDIMENSIONAL ARRAYS

# ### Creating array out of list

a = np.array([1, 2, 3])
print(a)

# a's type is numpy.ndarray (n-dimensional array)
print(type(a))

# the data type of the a's elements
print(a.dtype)

# a's shape (a 1-dimensional array of shape (3,) - 3 elements on one row)
print(a.shape)

# the first element of a
# a can be indexed as a list
print(a[0])

b = np.array([[1, 2, 3], [4, 5, 6]])

# b's shape is (2, 3) - 2 rows, 3 elements each
print(b.shape)

# b can be indexed as a normal list
print(b[0][2])

# specific indexing - equivalent with b[0][2]
print(b[0, 2])

# transforms any input data in ndarray
c = np.asarray([(1, 2), (3, 4)])

print(type(c))
print(c.shape)
print(c)


# ### Creating array using numpy functions

# creating an ndarray of a given shape full of 0
zero_array = np.zeros((3, 2))
print(zero_array)

# creating an ndarray of a given shape full of 1
ones_array = np.ones((2, 2))
print(ones_array)

# creating an ndarray of a given shape full of a given value
constant_array = np.full((2, 2), 8)
print(constant_array)

# creates the identity matrix of the given size
identity_matrix = np.eye(3)
print(identity_matrix)

# creates a ndarray with random values
# evenly distributed between [0, 1)
random_array = np.random.random((1, 2))
print(random_array)

mu, sigma = 1, 0.1
# creates a ndarray with random values of
# Gaussian distribution with mean mu and standard deviation sigma
gaussian_random = np.random.normal(mu, sigma, (3, 6))
print(gaussian_random)

# creates a ndarray of the first n natural values
first_5 = np.arange(5)
print(first_5)
