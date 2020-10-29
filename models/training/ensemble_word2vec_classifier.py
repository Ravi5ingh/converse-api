import utility.util as ut
import sklearn.neural_network as nn
import models.nl.processor as nl
import pipetools as pt
import numpy as np

class EnsembleWord2VecClassifier:
    """
    This is a simple classifier that combines word2vec, neural networks, and an ensemble technique (weighted voting)
    """

    def __init__(self, train_csv):
        """
        .ctor
        :param train_csv: The one-hot-encoded training CSV file name
        """

        self.__train_df__ = ut.read_csv(train_csv)

    def fit(self):
        """
        Fit the model
        """

        X, Y = self.__vectorize__()

        clf = nn.MLPClassifier(hidden_layer_sizes=(2),
                                     random_state=1,
                                     max_iter=10000)

        X = np.array(X)
        Y = np.array(Y, dtype=np.float)

        clf.fit(X, Y)

    def predict(self, text):
        """
        Given a piece of raw text, infer the intent of the user
        :return: The intent code
        """

    def __vectorize__(self):
        """
        Vectorizes the one-hot-encoded training data
        :return: The vectorized trainable data (X, Y)
        """

        X = []
        Y = []

        for index, row in self.__train_df__.iterrows():
            for word in self.__tokenize_text__(row['query']):
                vector, success = ut.try_word2vec(word)
                if success:
                    X.append(vector)
                    Y.append(row[1:].to_numpy())

        return X, Y

    def __tokenize_text__(self, text):
        """
        Take the raw text string and tokenize to a standardized string array
        :param text: The raw text
        :return: The tokenized text
        """

        return (pt.pipe
                | nl.normalize_text
                | nl.tokenize_text
                | nl.remove_stopwords
                | nl.lemmatize_text)(text)