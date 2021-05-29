# # Car Price Prediction
#
# Next, we are going to work with Car Price Prediction database, to predict the
# price of a car based on its characteristics.
#
# The database contains 4879 training examples. Because we don't have a testing
# dataset, we will use the cross validation technique to validate the parameters
# of the methods we are going to use.
#
# The first 4 training examples can be seen in the table below:
#
# Year | Kilometers_driven | Fuel_type | Transmission | Owner_type |   Mileage  | Engine  |   Power   | Seats | Price |
# :---:|:-----------------:|:----------|:-------------|:-----------|:----------:|:-------:|:---------:|:-----:|------:|
# 2010 |       72000       |    CNG    |    Manual    |    First   | 26.6 km/kg | 998 CC  | 58.16 bhp |   5   | 1.75  |
# 2012 |       87000       |   Diesel  |    Manual    |    First   | 20.77 kmpl | 1248 CC | 88.76 bhp |   7   |   6   |
# 2013 |       40670       |   Diesel  |   Automatic  |    Second  | 15.2 kmpl  | 1968 CC | 140.8 bhp |   5   | 17.74 |
# 2012 |       75000       |    LPG    |    Manual    |    First   | 21.1 km/kg | 814 CC  | 55.2 bhp  |   5   | 2.35  |
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge

plt.style.use('default')


# load training data
training_data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')

# print the first 4 samples
print('The first 4 samples are:\n ', training_data[:4])
print('The first 4 prices are:\n ', prices[:4])

# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)


# ### Exercise 1
#
# Define a function that takes as parameter the training data
# and returns the scaled data.

def normalize_data(train_data):
    scaler = StandardScaler()
    scaler.fit(train_data)

    scaled_train_data = scaler.transform(train_data)

    return scaled_train_data


# ### Exercise 2
#
# Using the training data from the Car Price Prediction dataset, train a linear regression model
# using cross-validation with 3 folds. Calculate the mean value of MSE and MAE.

s_train_x = normalize_data(training_data)

model = LinearRegression()
results_cv = cross_validate(model, s_train_x, prices, cv=3,
                            scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))
results_cv

print(np.abs(results_cv['test_neg_mean_squared_error'].mean()))
print(np.abs(results_cv['test_neg_mean_absolute_error'].mean()))


# ### Exercise 3
#
# Using the training data from the Car Price Prediction dataset, train a Ridge regression model
# using cross-validation with 3 folds. Calculate the mean value of MSE and MAE. Check which
# alpha $\in$ {0.1, 1, 10, 100, 1000} gets the best performance.

alphas = [0.1, 1, 10, 100, 1000]
for alpha in alphas:
    print('alpha =', alpha)
    ridge = Ridge(alpha=alpha)
    results = cross_validate(ridge, s_train_x, prices, cv=3,
                             scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'))
    print(np.abs(results['test_neg_mean_squared_error'].mean()))
    print(np.abs(results['test_neg_mean_absolute_error'].mean()))
    print()


# ### Exercise 4
#
# Using the best alpha found at Exercise 3, train a Ridge regression model on the entire
# training dataset and print the coefficients and the bias of the regression. Which is the most
# significant feature? What about the second? Which is the least significant feature?

ridge = Ridge(alpha=1)
ridge.fit(s_train_x, prices)

print(ridge.intercept_)    # the bias
print(ridge.coef_)         # the coefficients

ordered_importance_indexes = np.argsort(np.abs(ridge.coef_))
# the least important features are the first
print(ordered_importance_indexes)

features = {0: 'year',
            1: 'kilometers driven',
            2: 'mileage',
            3: 'engine',
            4: 'power',
            5: 'seats',
            6: 'owner type',        # values between 1 and 4
            7: 'fuel_type_1',
            8: 'fuel_type_2',
            9: 'fuel_type_3',
            10: 'fuel_type_4',
            11: 'fuel_type_5',
            12: 'transmission_type_1 (manual)',
            13: 'transmission_type_2 (automatic)',
            }

print('Most significant feature: ', features[ordered_importance_indexes[-1]])
print('Second most significant feature: ',
      features[ordered_importance_indexes[-2]])
print('Least significant feature: ', features[ordered_importance_indexes[0]])


# ### Visualizing the training data

print(features[0] + '\n', np.take(training_data, [0], 1))
print()
print(features[4] + '\n', np.take(training_data, [4], 1))
print()
print(features[2] + '\n', np.take(training_data, [2], 1))
print()
print(features[3] + '\n', np.take(training_data, [3], 1))


def scatter(x, feature, fig):
    plt.subplot(5, 2, fig)
    plt.scatter(np.take(training_data, [x], 1),
                prices, c='yellow', edgecolors='black')
    plt.title(feature.capitalize() + ' vs Price')
    plt.ylabel('Price')
    plt.xlabel(feature.capitalize())


plt.figure(figsize=(10, 20))

scatter(0, features[0], 1)
scatter(4, features[4], 2)
scatter(2, features[2], 3)
scatter(3, features[3], 4)

plt.tight_layout()
