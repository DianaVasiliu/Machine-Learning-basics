from sklearn import preprocessing
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


# # Data normalization

# ## Standardization
#
# Standardization transforms the feature arrays so their mean is 0 and standard deviation is 1
#
# The transformation formula is:
#
# ## $ x_{scaled} = \frac{x-mean(x)}{\sigma} $
#
# where $mean(x)$ is the mean of the values in x, and $\sigma$ is the standard deviation

x_train = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]], dtype=np.float64)
x_test = np.array([[-1, 1, 0]], dtype=np.float64)

# calculating statistics on the train data
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)

# printing the mean and the standard deviation
print(scaler.mean_)
print(scaler.scale_)

# scaling the training data
scaled_x_train = scaler.transform(x_train)
print(scaled_x_train)

# scaling the testing data
scaled_x_test = scaler.transform(x_test)
print(scaled_x_test)


# ## Normalization
#
# We will study the L1 and L2 normalizations.
#
# Normalization means transforming the feature arrays individually so their norm becomes 1.
#
# The formulas used for normalization are:
#
# ## $L_1 norm: x_{scaled} = \frac{X}{||X||_1}$
# ## $L_2 norm: x_{scaled} = \frac{X}{||X||_2}$
#
# where:
#
# $||X||_1 = \sum_{i=1}^{n}|x_i|$
#
# $||X||_2 = \sqrt{\sum_{i=1}^{n}x_i^2}$
