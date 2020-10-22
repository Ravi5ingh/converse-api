import utility.util as ut
import sklearn.pipeline as pi
import sklearn.model_selection as ms
import sklearn.neural_network as nn
import sklearn.metrics as me
import models.pipelines.transformers.google_word_vectorizer as go
import global_variables as gl
import numpy as np

class NeuralWord2VecPipeline:
    """
    This class represents the pipeline that houses models that work by vectorizing the tweets using Word2Vec and then
    training to the output categories using Neural Networks
    """

    def __init__(self, disaster_db_file, disaster_table_name):

        self.__disaster__ = ut.read_db(disaster_db_file, disaster_table_name)

    def init_fit_eval(self):
        """
        Build a pipeline and fit it with GridSearchCV
        """

        pipeline, x_train, x_test, y_train, y_test = self.__build_pipeline__()

        self.__perform_gridsearch__(
            pipeline,
            x_train,
            x_test,
            y_train,
            y_test,
            parameters={
                'clf__solver': ['sgd', 'lbfgs']
            }
        )

    def predict(self, tweet):
        """
        Classify the tweet in terms of the categories concerned
        :param tweet: The raw tweet
        :return: The classifications for each concerned category
        """

        predictions = {}
        i = 0
        for category in gl.disaster_response_target_columns:
            predictions[category] = self.__grid_searcher__.predict([tweet])[0][i]
            i += 1

        return predictions

    def __build_pipeline__(self):
        """
        Build the pipeline, and split training data
        :return: The pipeline and the split up training data
        """

        print('Create Neural Word2Vec Pipeline...')

        X = self.__disaster__['message'].values
        Y = self.__disaster__[gl.disaster_response_target_columns].values

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

    def __perform_gridsearch__(self, pipeline, x_train, x_test, y_train, y_test, parameters):
        """
        Perform grid search on training data, with the provided pipeline and the grid search parameters
        :param pipeline: The pipeline
        :param x_train: X Train
        :param x_test: X Test
        :param y_train: Y Train
        :param y_test: Y Test
        :param parameters: Grid Search Parameters
        """

        self.__grid_searcher__ = ms.GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

        self.__grid_searcher__.fit(x_train, y_train)

        y_pred = self.__grid_searcher__.predict(x_test)

        try:
            self.__print_summary__(y_test, y_pred)
        except:
            print('An unexpected error occurred while trying to print summary. Carrying on...')

        print('Performed grid search. Accuracy: ' + str(self.__get_accuracy__(y_test, y_pred)))

        best_parameters = self.__grid_searcher__.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def __print_summary__(self, y_test, y_pred):
        """
        Print the summary for each category
        :return:
        """

        # Transpose values to enable category iteration
        y_test = np.array(y_test).T
        y_pred = np.array(y_pred).T

        # Print confusion matrix for each category
        for i in range(0, len(gl.disaster_response_target_columns)):

            try:
                print('Confusion Matrix for ' + str(gl.disaster_response_target_columns[i]))
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



