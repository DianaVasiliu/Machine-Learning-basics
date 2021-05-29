import numpy as np

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])

# adding each element of an array (v) to each row of a matrix (m)
y = m + v
print(y)


# ## Broadcasting rules:
# 1. if the arrays don't have the same dimensions, then the smaller array
#    is extended by one dimension, until they have the same dimensions
#         a.shape = (3, 4)
#         b.shape = (6,)
#         => b is extended to dimension (6, 1)
#
# 2. two arrays are COMPATIBLE on one dimension
#    if they have the same length on that dimension
#    or if one of them has length 1
#         a.shape = (3, 4)
#         b.shape = (6, 1)
#         c.shape = (3, 5)
#         => a and c are compatible on the first dimension
#            a and b are compatible on the second dimension
#
# 3. broadcasting can be applied on two arrays only if
#    they are compatible on all their dimensions
#
# 4. at broadcasting, each array acts like they have the maximum dimension
#    of all arrays, on all their dimensions
#         a.shape = (3, 4)
#         b.shape = (3, 1)
#         => they both act like they are (3, 4)
#    and b acts like it's copied along that dimension
