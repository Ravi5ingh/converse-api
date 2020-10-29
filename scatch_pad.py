import utility.util as ut
import models.training.ensemble_word2vec_classifier as ec
import pandas as pd
import numpy as np
import sklearn.neural_network as nn

ut.widen_df_display()

# X = []
# avg = np.average(
#     [np.array([1, 2, 3, 4]),
#     np.array([1, 2, 3, 4]),
#     np.array([1, 2, 3, 4])],
#     axis=0
# )
# credit_vect, s = ut.try_word2vec('credit')
#
# ut.whats(X)
# ut.whats(credit_vect)
# ut.whats(avg)

# clf = ec.EnsembleWord2VecClassifier('models/train.csv')
#
# clf.fit()

# X = np.arange(33000).reshape(110,300)
# Y = np.zeros(330).reshape(110,3)
#
# clf = nn.MLPClassifier(hidden_layer_sizes=(10),
#                        random_state=1,
#                        max_iter=1000)
#
# clf.fit(X, Y)

classifier = ec.EnsembleWord2VecClassifier('models/train.csv')

classifier.fit()

# train = ut.read_csv('models/train.csv')
#
# print(train['is_credit_limit'])