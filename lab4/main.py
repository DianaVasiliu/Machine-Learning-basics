import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# # SMS Spam Classification
#
# The database contains SMS messages which are either spam or non-spam. The goal is to classify a message as spam or non-spam.
#
# The database contains 3734 training examples and 1840 testing examples. The non-spam:spam ratio is 6:1.
#
# Examples of sentences in the dataset:
#
# **spam** URGENT! We are trying to contact you. Last weekends draw shows that you have
# won a £900 prize GUARANTEED. Call 09061701939. Claim code S89. Valid 12hrs only
#
# **ham** Hi frnd, which is best way to avoid missunderstding wit our beloved one's?
#
# **ham** Great escape. I fancy the bridge but needs her lager. See you tomo

# ## Loading data

train_sentences = np.load(
    'data_lab4/data/training_sentences.npy', allow_pickle=True)
train_labels = np.load('data_lab4/data/training_labels.npy')
test_sentences = np.load(
    'data_lab4/data/test_sentences.npy', allow_pickle=True)
test_labels = np.load('data_lab4/data/test_labels.npy')


# ## EXERCISES
#
# ### 2. Define a function `normalize_data(train_data, test_data, type=None)` having the parameters as follows: the training data, the testing data and the type of normalization (can be one of `{None, standard, l1, l2}`) and returns the normalized data.

def normalize_data(train_data, test_data, type=None):
    if type is None:
        return train_data, test_data

    if type == 'l1' or type == 'l2':
        scaler = Normalizer(norm=type)
    elif type == 'standard':
        scaler = StandardScaler()
    elif type == 'minmax':
        scaler = MinMaxScaler()
    else:
        return train_data, test_data

    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    return scaled_train_data, scaled_test_data


# ### 3. Define the class BagOfWords whose constructor initializes the vocabulary (an empty dictionary).
#
# Inside the class, define the method `build_vocabulary(self, data)`, whose `data` parameter is a list of messages (a list of list of strings) and constructs the vocabulary based on the data.
#
# The dictionary's keys are the words and the values are unique IDs for each word.
#
# Further more, make a list of words inside the class, which contains the words in the reading order.
#
# Print the vocabulary length (must be 9522).
#
# **Note:** The vocabulary will be constructed based only on the training data.
#
# ### 4. Define the method `get_features(self, data)` whose `data` parameter is a list of messages of dimension `num_samples` and returns a matrix of dimension `num_samples * dictionary_length` defined as:
#
# $features(sample_{idx}, word_{idx})$ = the number of apparitions of the word whose ID is $word_{idx}$ in the document (message) $sample_{idx}$.

class BagOfWords:
    def __init__(self):
        self.vocabulary = dict()
        self.words = list()

    # ex3
    def build_vocabulary(self, data):
        self.vocabulary['UNK'] = 0
        i = 1
        for sentence in data:
            for word in sentence:
                if word not in self.vocabulary:
                    self.vocabulary[word] = i
                    self.words.append(word)
                    i += 1

        print(len(self.vocabulary))

    # ex4
    def get_features(self, data):
        features = np.zeros((len(data), len(self.vocabulary)))
        for idx, sentence in enumerate(data):
            for word in sentence:
                if word in self.vocabulary:
                    features[idx, self.vocabulary[word]] += 1
                else:
                    features[idx, 0] += 1
        return features


# ### 5. Using the functions defined above, calculate the BOW representation for the training data and testing data, then normalize them using L2 norm.

bag = BagOfWords()
bag.build_vocabulary(train_sentences)
train_features = bag.get_features(train_sentences)
test_features = bag.get_features(test_sentences)
scaled_train_sentences, scaled_test_sentences = normalize_data(
    train_features, test_features, type='l2')

# testing
print(scaled_train_sentences[:1])
print(scaled_test_sentences[:1])


# ### 6. Train a SVM with linear kernel that classifies spam/non-spam messages. Use parameter C of value 1.
#
# Calculate the `accuracy` and `F1-score` for the testing data.

svm_model = SVC()
svm_model.C = 1
svm_model.kernel = 'linear'

# train - nonscaled data
svm_model.fit(train_features, train_labels)

# predict - nonscaled data
predictions = svm_model.predict(test_features)
print(accuracy_score(test_labels, predictions))
print(f1_score(test_labels, predictions))

svm_model = SVC()
svm_model.C = 1
svm_model.kernel = 'linear'

# train - scaled data
svm_model.fit(scaled_train_sentences, train_labels)

# predict - scaled data
predictions = svm_model.predict(scaled_test_sentences)
print(accuracy_score(test_labels, predictions))
print(f1_score(test_labels, predictions))


# ### Print the 10 most negative (spam) words and 10 most positive (non-spam) words.

# Extra
indexes = np.argsort(svm_model.coef_[0])

nonspam = [bag.words[x - 1] for x in indexes[:10]]
print(nonspam)

spam = [bag.words[x - 1] for x in indexes[-10:]]
print(spam)
