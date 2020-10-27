import utility.util as ut
import sklearn.pipeline as pi
import sklearn.model_selection as ms
import sklearn.neural_network as nn
import sklearn.metrics as me
import models.training.transformers.google_word_vectorizer as go
import numpy as np

class NeuralWord2VecPipeline:
    """
    This class represents the pipeline that houses models that work by vectorizing the queries using Word2Vec and then
    training to the output categories using Neural Networks
    """

    def __init__(self, train_csv):
        """
        .ctor
        :param train_csv: The CSV file with the training data
        """

        self.__train_df__ = ut.read_csv(train_csv)
        self.__target_columns__ = [col for col in self.__train_df__.columns if col not in ['query']]

    def init_fit_eval(self):
        """
        Build a pipeline and fit it with GridSearchCV
        """

        self.__pipeline__, x_train, x_test, y_train, y_test = self.__build_pipeline__()

        self.__pipeline__.fit(x_train, y_train)

    def predict(self, query):
        """
        Classify the query in terms of the categories concerned
        :param query: The raw query
        :return: The classifications for each concerned category
        """

        raw_predictions = self.__pipeline__.predict([query])
        predictions = {}
        i = 0
        for category in self.__target_columns__:
            predictions[category] = raw_predictions[0][i]
            i += 1

        return predictions

    def __build_pipeline__(self):
        """
        Build the pipeline, and split training data
        :return: The pipeline and the split up training data
        """

        print('Create Neural Word2Vec Pipeline...')

        X = self.__train_df__['query'].values
        Y = self.__train_df__[self.__target_columns__].values

        x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.25)

        return pi.Pipeline([
            ('vect', go.GoogleWordVectorizer()),
            ('clf', nn.MLPClassifier(hidden_layer_sizes=(36),
                                     random_state=1,
                                     max_iter=1000))
        ]),\
        x_train,\
        x_test,\
        y_train,\
        y_test

    def __print_summary__(self, y_test, y_pred):
        """
        Print the summary for each category
        :return:
        """

        # Transpose values to enable category iteration
        y_test = np.array(y_test).T
        y_pred = np.array(y_pred).T

        # Print confusion matrix for each category
        for i in range(0, len(self.__target_columns__)):

            try:
                print('Confusion Matrix for ' + str(self.__target_columns__[i]))
                conf_matrix = me.confusion_matrix(y_test[i], y_pred[i])
                print(me.confusion_matrix(y_test[i], y_pred[i]))
                if len(conf_matrix) > 1:
                    precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])
                    recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
                    print('Precision: ' + str(precision))
                    print('Recall: ' + str(recall))
                    print('F1 Score: ' + str(2 * ((precision * recall) / (precision + recall))))
                print('-------------')
            except:
                print('Error occurred while trying to print summary for category at index ' + str(i))

        pass

    def __get_accuracy__(self, y_test, y_pred):
        """
        Get the accuracy given multi output actual vs predicted values
        Accuracy = Correctly predicted values / All values
        :param y_test: The actual values
        :param y_pred: The predicted values
        """

        i = 0
        num_correct = 0
        for prediction in y_pred:
            num_correct += sum(
                map(
                    lambda pair: 1 if ((pair[0] + pair[1] == 2) or (pair[0] + pair[1] == 0))
                    else 0,
                    zip(y_test[i], y_pred[i])))
            i += 1

        return num_correct / (len(y_pred) * len(y_pred[0]))



