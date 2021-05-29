import numpy as np

# ## Common slicing

array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# getting all rows (:) and the first 3 elements on each row ([0:3])
array_slice = array[:, 0:3]
print(array_slice)

# modifying the slice also modifies the initial array
print("Pay attention to modifications!")
print(array[0][0])
array_slice[0][0] = 100
print(array[0][0])

print("Let's repair this bug")
array[0][0] = 1
slice_copy = np.copy(array[:, 0:3])
slice_copy[0][0] = 100
print(slice_copy[0][0])
print(array[0][0])

# if one index is an integer, the slice shape is smaller than the original
slice1 = array[2:3, :]
print(slice1)
slice2 = array[2, :]
print(slice2)

# creating 1D array from ndarray
array_1d = np.ravel(array)
print(array_1d)

# reshaping ndarray
# all the elements must fit in the new shape
reshaped_array = np.reshape(array, (2, 6))
print(reshaped_array)


# ## Integer lists slicing

# printing the specified positions in the array
# printing elements on (0, 1) and (0, 3)
# printing elements on positions obtained by
# calculating the cartesian product of the lists
print(array)
print(array[[0, 0], [1, 3]])


# ## Boolean value slicing

# creates a ndarray of the same shape of the original array
# where each value is transformed in True or False
# according to the given condition
# in this case, bool_idx[i][j] is True if array[i][j] > 10
# and False otherwise
bool_idx = (array > 10)
print(bool_idx)

# another ways to use the boolean value slicing
print(array[bool_idx])
print(array[array > 10])
