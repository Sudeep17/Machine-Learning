import sys
import os
import numpy as np
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

lemmatizer = WordNetLemmatizer()


def read_file(folder):
    """
    returns the contents of files present under the folder as list
    :param folder: directory where files are present
    :return: returns the contents of files present under the folder as list
    """
    a_list = []
    file_list = os.listdir(folder)
    for file in file_list:
        f = open(folder + file, 'r', encoding='Latin-1')
        a_list.append(f.read())
    f.close()
    return a_list


def find_prob(arr, val, no_of_words):
    """
    extract probability of a word present in the array
    :param arr: array to be searched
    :param val: keyword to be searched
    :param no_of_words: number of words in array
    :return: probability of the keyword if found else returns np.log(1/no_of_words)
    """
    if len(np.where(arr[0] == val, )[0]):
        return arr[1][np.where(arr[0] == val, )[0][0]]
    else:
        return np.log(1/no_of_words)


def load_data(train, test):
    """
    labels train and test data
    :param train: training data set
    :param test: test data set
    :returns: list containing the content of email as well as the label for it for both training and test data
    """
    spam_train = read_file(train + "/spam/")
    ham_train = read_file(train + "/ham/")
    spam_test = read_file(test + "/spam/")
    ham_test = read_file(test + "/ham/")

    spam_emails_train = [(email, 'spam') for email in spam_train]
    ham_emails_train = [(email, 'ham') for email in ham_train]
    all_emails_train = spam_emails_train + ham_emails_train
    # random.shuffle(all_emails_train)

    spam_emails_test = [(email, 'spam') for email in spam_test]
    ham_emails_test = [(email, 'ham') for email in ham_test]
    all_emails_test = spam_emails_test + ham_emails_test
    # random.shuffle(all_emails_test)

    return all_emails_train, all_emails_test


def pre_process(sentence, remove_stop_words, nlp_tokenizer):
    """
    Extract token from sentence
    :param sentence: sentence to be tokenized
    :param remove_stop_words: flag to remove stop word from sentence if set to True else not
    :param nlp_tokenizer: flag to use nltk tokenizer if set to True else split the sentence by whitespace
    :return: list of words
    """
    if nlp_tokenizer and nlp_tokenizer == 'True':
        tokens = word_tokenize(sentence)
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    else:
        tokens = sentence.lower().split(' ')
    if remove_stop_words:
        tokens = [t for t in tokens if t not in stop_word_list and len(t) > 2]
    return tokens


def extract_feature(data, remove_stop_words, nlp_tokenizer):
    """
    calculate the conditional probability of word present in the data given a class
    :param data: data to extract feature from
    :param remove_stop_words: flag to remove stop word from sentence if set to True else not
    :param nlp_tokenizer: flag to use nltk tokenizer if set to True else split the sentence by whitespace
    :return: conditional probability of word for the vocabulary present for the data grouped by class
    """
    feature_spam = []
    feature_ham = []
    for (email_body, email_label) in data:
        if email_label == 'spam':
            feature_spam.extend(pre_process(email_body, remove_stop_words, nlp_tokenizer))

        else:
            feature_ham.extend(pre_process(email_body, remove_stop_words, nlp_tokenizer))
    spam_word = np.shape(feature_spam)
    ham_word = np.shape(feature_ham)
    word, count = np.unique(feature_spam, return_counts=True)
    feature_spam = np.asarray((word, count)).T
    spam_word_unique = np.shape(feature_spam)
    word, count = np.unique(feature_ham, return_counts=True)
    feature_ham = np.asarray((word, count)).T
    ham_word_unique = np.shape(feature_ham)
    for i in range(np.shape(feature_spam)[0]):
        feature_spam[i][1] = np.log((int(feature_spam[i][1]) + 1) / (spam_word[0] + spam_word_unique[0]))
    for i in range(np.shape(feature_ham)[0]):
        feature_ham[i][1] = np.log((int(feature_ham[i][1]) + 1) / (ham_word[0] + ham_word_unique[0]))
    return feature_spam.T, feature_ham.T, spam_word[0], ham_word[0]


def naive_bayes_classifier(features_spam, features_ham, test_data, spam_word_count, ham_word_count, remove_stop_words, nlp_tokenizer):
    """
    Uses Naive Bayes Classifier to train data and evaluate the accuracy on test data
    :param features_spam: feature extracted for spam mails
    :param features_ham: feature extracted for ham mails
    :param test_data: test data for testing the classifier
    :param spam_word_count: vocabulary strength for spam mails
    :param ham_word_count: vocabulary strength for ham mails
    :param remove_stop_words: flag to remove stop word from sentence if set to True else not
    :param nlp_tokenizer: flag to use nltk tokenizer if set to True else split the sentence by whitespace
    :return: accuracy on test data
    """
    pos_case = 0
    neg_case = 0
    for (body, label) in test_data:
        cond_prob_spam = float(np.log(130 / 478))
        cond_prob_ham = float(np.log(348 / 478))
        word_list = pre_process(body, remove_stop_words, nlp_tokenizer)
        for w in word_list:
            p1 = float(find_prob(features_spam, w, spam_word_count))
            p2 = float(find_prob(features_ham, w, ham_word_count))
            cond_prob_spam = cond_prob_spam + p1
            cond_prob_ham = cond_prob_ham + p2

        # print(cond_prob_spam, cond_prob_ham, cond_prob_spam > cond_prob_ham, label)
        if cond_prob_spam >= cond_prob_ham and label == 'spam':
            pos_case = pos_case + 1
        elif cond_prob_spam <= cond_prob_ham and label == 'ham':
            pos_case = pos_case + 1
        else:
            neg_case = neg_case + 1
    acc = (pos_case / (pos_case + neg_case))*100
    return acc

# ------------------------ Main Block ----------------------- #
if sys.argv.__len__() >= 4:

    nltk.download('stopwords')
    stop_word_list = stopwords.words("english")
    print()
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    nlp_tokenizer = sys.argv[3]
    remove_stop_word = False

    print('Loading Data')
    training_data, test_data = load_data(train_data_path, test_data_path)
    print()

    print('Training Naive Bayes Classifier (stopword not removed) .....')
    features_spam_train, feature_ham_train, spam_word_count, ham_word_count = extract_feature(training_data, remove_stop_word, nlp_tokenizer)
    print('Naive Bayes Classifier Trained')
    print('Evaluating accuracy of Naive Bayes (stopword not removed) .....')
    accuracy = naive_bayes_classifier(features_spam_train, feature_ham_train, test_data, spam_word_count, ham_word_count, remove_stop_word, nlp_tokenizer)
    print('Accuracy of test data without removing stopwords using Naive Bayes Classifier (stopword not removed):', np.round(accuracy, 3), '%')

    print()
    print('Training Naive Bayes Classifier (stopword removed) .....')
    features_spam_train, feature_ham_train, spam_word_count, ham_word_count = extract_feature(training_data, not remove_stop_word, nlp_tokenizer)
    print('Naive Bayes Classifier Trained')
    print('Evaluating accuracy of Naive Bayes (stopword removed) .....')
    accuracy = naive_bayes_classifier(features_spam_train, feature_ham_train, test_data, spam_word_count, ham_word_count, not remove_stop_word, nlp_tokenizer)

    print('Accuracy of test data using Naive Bayes Classifier (stopword removed):', np.round(accuracy, 3), '%')

else:
    print("Invalid number of arguments. Arguments required:",
          "<training-set> <test-set> <nltk_tokenizer>")
