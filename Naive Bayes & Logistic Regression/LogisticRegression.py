import os
import sys
import collections
import re
import math
from nltk.corpus import stopwords
import nltk


# Stores emails as dictionaries. email_file_name : Document (class defined below)
training_set = dict()
test_set = dict()

# Vocabulary/tokens in the training set
training_set_vocab = []

# store weights as dictionary. w0 initiall 0.0, others initially 0.0. token : weight value
weights = {'weight_zero': 0.0}

# ham = 0 for not spam, spam = 1 for is spam
classes = ["ham", "spam"]

# Natural learning rate constant, number of iterations for learning weights, and penalty (lambda) constant
learning_constant = .01
penalty = 0.0


def make_data_set(storage_dict, directory, true_class):
    """
    Read all text files in the given directory and construct the data set, D
    the directory path should just be like "train/ham" for example
    storage is the dictionary to store the email in
    True class is the true classification of the email (spam or ham)
    """
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, encoding="Latin-1") as text_file:
                # stores dictionary of dictionary of dictionary as explained above in the initialization
                text = text_file.read()
                storage_dict.update({dir_entry_path: Document(text, bag_of_words(text), true_class)})


def extract_vocab(data_set):
    """
    Extracts the vocabulary of all the text in a data set
    :param data_set: data set
    :return: array containing the vocabulary
    """
    v = []
    for i in data_set:
        for j in data_set[i].getWordFreqs():
            if j not in v:
                v.append(j)
    return v


def bag_of_words(text):
    """
    counts frequency of each word in the text files and order of sequence doesn't matter
    """
    bagsofwords = collections.Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)


def learn_weights(training, weights_param, iterations, lam):
    """
    Learn weights by using gradient ascent
    """
    # Adjust weights num_iterations times
    for x in range(0, iterations):
        # print(x)
        # Adjust each weight...
        counter = 1
        for w in weights_param.copy():
            sum = 0.0
            # ...using all training instances
            for i in training:
                # y_sample is true y value (classification) of the doc
                y_sample = 0.0
                if training[i].getTrueClass() == classes[1]:
                    y_sample = 1.0
                # Only add to the sum if the doc contains the token (the count of it would be 0 anyways)
                if w in training[i].getWordFreqs():
                    sum += float(training[i].getWordFreqs()[w]) * (y_sample - calculate_cond_prob(classes[1], weights_param, training[i]))
            weights_param[w] += ((learning_constant * sum) - (learning_constant * float(lam) * weights_param[w]))


def calculate_cond_prob(class_prob, weights_param, doc):
    """ Calculate conditional probability for the specified doc. Where class_prob is 1|X or 0|X
    1 is spam and 0 is ham
    """
    # Total tokens in doc. Used to normalize word counts to stay within 0 and 1 for avoiding overflow
    # total_tokens = 0.0
    # for i in doc.getWordFreqs():
    #     total_tokens += doc.getWordFreqs()[i]

    # Handle 0
    if class_prob == classes[0]:
        sum_wx_0 = weights_param['weight_zero']
        for i in doc.getWordFreqs():
            if i not in weights_param:
                weights_param[i] = 0.0
            # sum of weights * token count for each token in each document
            sum_wx_0 += weights_param[i] * float(doc.getWordFreqs()[i])
        if sum_wx_0 > 500:
            return 1.0 / (1.0 + math.exp(float(500)))
        return 1.0 / (1.0 + math.exp(float(sum_wx_0)))

    # Handle 1
    elif class_prob == classes[1]:
        sum_wx_1 = weights_param['weight_zero']
        for i in doc.getWordFreqs():
            if i not in weights_param:
                weights_param[i] = 0.0
            # sum of weights * token count for each token in each document
            sum_wx_1 += weights_param[i] * float(doc.getWordFreqs()[i])
        if sum_wx_1 > 500:
            return math.exp(float(500)) / (1.0 + math.exp(float(500)))
        return math.exp(float(sum_wx_1)) / (1.0 + math.exp(float(sum_wx_1)))


def apply_logistic_regression(data_instance, weights_param):
    """
    Apply algorithm to guess class for specific instance of test set
    """
    score = {}
    score[0] = calculate_cond_prob(classes[0], weights_param, data_instance)
    score[1] = calculate_cond_prob(classes[1], weights_param, data_instance)
    if score[1] > score[0]:
        return classes[1]
    else:
        return classes[0]


# Document class to store email instances easier
class Document:
    text = ""
    # x0 assumed 1 for all documents (training examples)
    word_freqs = {'weight_zero': 1.0}

    # spam or ham
    true_class = ""
    learned_class = ""

    # Constructor
    def __init__(self, text, counter, true_class):
        self.text = text
        self.word_freqs = counter
        self.true_class = true_class

    def getText(self):
        return self.text

    def getWordFreqs(self):
        return self.word_freqs

    def getTrueClass(self):
        return self.true_class

    def getLearnedClass(self):
        return self.learned_class

    def setLearnedClass(self, guess):
        self.learned_class = guess


# takes directories holding the data text files as paramters. "train/ham" for example
def main(trainingdatadir, testdatadir, lambda_constant, no_of_iteration, remove_stopWords):
    # Set up data sets. Dictionaries containing the text, word frequencies, and true/learned classifications
    training_spam_dir, training_ham_dir, test_spam_dir, test_ham_dir = trainingdatadir+"/spam" , trainingdatadir+"/ham", testdatadir+"/spam", testdatadir+"/spam"

    print('Loading Data')
    traindata ={}
    spamDataSet={}
    make_data_set(spamDataSet, training_spam_dir, classes[1])
    hamDataSet={}
    make_data_set(hamDataSet, training_ham_dir, classes[0])



    splitSpamDataSet = dict(list(spamDataSet.items())[:int((len(spamDataSet) / 100) * 70)])
    splitHamDataSet = dict(list(hamDataSet.items())[:int((len(hamDataSet) / 100) * 70)])
    splitSpamDataSet.update(splitHamDataSet)
    splitTraining_set =splitSpamDataSet.copy()
    splitValidationSet ={}

    splitSpamDataSet = dict(list(spamDataSet.items())[int((len(spamDataSet) / 100) * 70):])
    splitHamDataSet = dict(list(hamDataSet.items())[int((len(hamDataSet) / 100) * 70):])
    traindata.update(splitSpamDataSet)

    traindata.update(splitHamDataSet)
    splitValidationSet = traindata.copy()


    lambdaEfficiency={}
    penalty = lambda_constant

    # Extract training set vocabulary
    training_set_vocab = extract_vocab(splitTraining_set)
    if remove_stopWords == 'true':
        nltk.download('stopwords')
        stop_word_list = stopwords.words("english")
        training_set_vocab = [t for t in training_set_vocab if t not in stop_word_list]


    print(len(training_set_vocab))
    # Set all weights in training set vocabulary to be initially 0.0. w0 ('weight_zero') is initially 0.0
    for i in training_set_vocab:
        weights[i] = 0.0

    while float(penalty) <= 1.0:
        print()
        print('Training Logistic Regression Classifier with lambda', str(penalty), '....')
        # Learn weights
        learn_weights(splitTraining_set, weights, no_of_iteration, penalty)

        # Apply algorithm on test set
        correct_guesses = 0.0
        for i in splitValidationSet:
            splitValidationSet[i].setLearnedClass(apply_logistic_regression(splitValidationSet[i], weights))
            if splitValidationSet[i].getLearnedClass() == splitValidationSet[i].getTrueClass():
                correct_guesses += 1.0

        print('Logistic Regression Classifier Trained')
        accuracy = (100.0 * float(correct_guesses) / float(len(splitValidationSet)))
        lambdaEfficiency[penalty] = accuracy
        print("Accuracy of test data using Logistic Regression Classifier : %.3f%%" % (100.0 * float(correct_guesses) / float(len(splitValidationSet))))

        penalty = float(penalty) + 0.2
    test_set={}
    make_data_set(test_set, testdatadir+"/spam", classes[1])
    make_data_set(test_set, testdatadir+"/ham", classes[0])
    correct_guesses = 0.0
    for i in test_set:
        test_set[i].setLearnedClass(apply_logistic_regression(test_set[i], weights))
        if test_set[i].getLearnedClass() == test_set[i].getTrueClass():
            correct_guesses += 1.0
    accuracy = (100.0 * float(correct_guesses) / float(len(test_set)))
    lambdaEfficiency[penalty] = accuracy
    print()
    print('Training Logistic Regression Classifier with lambda', str(penalty), '....')
    print ("Accuracy of test data using Logistic Regression Classifier : %.3f%%" % (100.0 * float(correct_guesses) / float(len(test_set))))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5] )