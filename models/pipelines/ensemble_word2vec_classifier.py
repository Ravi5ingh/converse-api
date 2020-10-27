import utility.util as ut

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

    def predict(self, text):
        """
        Given a piece of raw text, infer the intent of the user
        :return: The intent code
        """