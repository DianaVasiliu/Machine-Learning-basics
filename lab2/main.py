import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


# # MNIST dataset

# ### Loading data

# loading all the necessary images
train_images = np.loadtxt('data_MNIST/data/train_images.txt')
test_images = np.loadtxt('data_MNIST/data/test_images.txt')
# loading the labels as integers
train_labels = np.loadtxt('data_MNIST/data/train_labels.txt', 'int')
test_labels = np.loadtxt('data_MNIST/data/test_labels.txt', 'int')

print(train_labels[:10])

# getting the first image
image = train_images[0, :]
# reshaping it so we can plot it
image = np.reshape(image, (28, 28))
plt.imshow(image.astype(np.uint8), cmap='gray')
plt.show()


# ## EXERCISES
# ### 2. Given that the minimum value of a pixel is 0 and the maximum value is 255, calculate the ends pf `num_bin` intervals (use `linspace` function).
# Define the function `values_to_bins` which expects a matrix of dimension `(n_samples, n_features)` and the interval heads computed before.
#
# For each data, calculate the index of the corresponding interval (use the `np.digitize` function).
#
# Use this function on the training data and testing data.

def values_to_bins(data, bin_array):
    data_binned = np.digitize(data, bin_array)
    return data_binned


# because our data (the pixel values) are continuous values,
# we have to transform them into discrete values
# so, we set the number of intervals we want to split the data into
num_bins = 5

# getting the intervals
bins = np.linspace(start=0, stop=255, num=num_bins)
print(bins)

# and then we assign the interval number to each continuous value
# using np.digitize(x, bins)
train_binned = values_to_bins(train_images, bins)
test_binned = values_to_bins(test_images, bins)
print(train_binned)
print(test_binned)

# Note: element indexing starts at 1, because there are no values < 0


# ### 3. Calculate the `MultinomialNB` accuracy on the testing data, splitting the pixel intervals in 5 subintervals.
# **Note:** the output accuracy on 5 bins must be `80.6%`.

# defining the Multinomial Naive Bayes model
nbModel = MultinomialNB()

# fitting the model according to train_binned, train_labels
# train_binned = the training data
# train_labels = target values
nbModel.fit(train_binned, train_labels)

# returning the accuracy given the testing data
nbModel.score(test_binned, test_labels)


# ### 4. Test the `MultinomialNB` using `num_bins` in `{3, 5, 7, 9, 11}`.

maxi = 0
maxScore = 0
for num_bins in [3, 5, 7, 9, 11]:
    nbModel = MultinomialNB()
    bins = np.linspace(start=0, stop=255, num=num_bins)

    train_binned = values_to_bins(train_images, bins)
    test_binned = values_to_bins(test_images, bins)

    nbModel.fit(train_binned, train_labels)
    score = nbModel.score(test_binned, test_labels)
    print(score)

    if score > maxScore:
        maxi = num_bins
        maxScore = score

# ### 5. Using the number of subintervals that obtains the best accuracy, show at least 10 misclassified examples.

nbModel = MultinomialNB()
num_bins = maxi
bins = np.linspace(start=0, stop=255, num=num_bins)

train_binned = values_to_bins(train_images, bins)
test_binned = values_to_bins(test_images, bins)

nbModel.fit(train_binned, train_labels)
score = nbModel.score(test_binned, test_labels)
print(score)

# getting the predicted output for testing data
predicted = nbModel.predict(test_binned)
# getting the misclassified images after the orediction
misclassified_images = test_images[predicted != test_labels]
misclassified_labels = predicted[predicted != test_labels]

# showing 10 misclassified images
for i in range(10):
    plt.imshow(misclassified_images[i].reshape(
        28, 28).astype(np.uint8), cmap='gray')
    plt.title(
        f'This image has been misclassified as {misclassified_labels[i]}')
    plt.show()


# ### 6. Define the function `confusion_matrix(y_true, y_pred)` that calculates the confusion matrix. Use the predictions obtained before.
# **Note:**
# `confusionMatrix[i][j]` = number of examples of class i that has been classified as j

c = np.zeros((10, 10))
for y_true, y_pred in zip(test_labels, predicted):
    c[y_true, y_pred] += 1

print(c)
