import utility.util as ut
import numpy as np
import sklearn.feature_extraction.text as te
import pipetools as pt
import models.nl.processor as nl

class GoogleWordVectorizer(te.BaseEstimator):
    """
    Vectorizes text based on Google's trained Word2Vec model.
    """

    def fit_transform(self, messages, y=None):
        """
        Transforms training data
        :param messages: The raw tweets
        :param y: y
        :return: Array of (1 vector for every message)
        """

        ret_val = []
        dimensionality = 0

        i = 0
        total = len(messages)
        for message in messages:
            words = self.__tokenize_query__(message)
            word_vectors = []
            for word in words:
                vector, success = ut.try_word2vec(word)
                if success:
                    dimensionality = dimensionality if dimensionality > 0 else len(vector)
                    word_vectors.append(vector)

            if word_vectors:
                ret_val.append(np.average(word_vectors, axis=0))
            else:
                ret_val.append([0] * dimensionality)

            i += 1
            if i%100==0 or i==total:
                ut.printover('(fit_transform) Vectorized ' + str(i) + ' of ' + str(total))

        print('\n')
        return np.asarray(ret_val)

    def fit(self, messages):
        """
        Necessary to comply with the template
        """
        pass

    def transform(self, messages):
        """
        Transforms validation data
        :param messages: The raw tweets
        :param y: y
        :return: Array of (1 vector for every message)
        """

        ret_val = []
        dimensionality = 0

        i = 0
        total = len(messages)
        for message in messages:
            words = self.__tokenize_query__(message)
            word_vectors = []
            for word in words:
                vector, success = ut.try_word2vec(word)
                if success:
                    dimensionality = dimensionality if dimensionality > 0 else len(vector)
                    word_vectors.append(vector)

            if word_vectors:
                ret_val.append(np.average(word_vectors, axis=0))
            else:
                ret_val.append([0] * dimensionality)

            i += 1
            if i%100==0 or i == total:
                ut.printover('(transform) Vectorized ' + str(i) + ' of ' + str(total))

        print('\n')
        return np.asarray(ret_val)

    def __tokenize_query__(self, query):
        """
        Take the raw query string and tokenize to a standardized string array
        :param query: The raw query
        :return: The tokenized query
        """

        return (pt.pipe
                | nl.normalize_text
                | nl.tokenize_text
                | nl.remove_stopwords
                | nl.lemmatize_text)(query)