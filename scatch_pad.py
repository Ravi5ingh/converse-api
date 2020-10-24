# import sklearn.pipeline as pi
# import models.pipelines.transformers.google_word_vectorizer as go
# import sklearn.neural_network as nn
#
# ss = pi.Pipeline([
#             ('SINGH', go.GoogleWordVectorizer()),
#             ('clf', nn.MLPClassifier(hidden_layer_sizes=(3),
#                                      random_state=1,
#                                      max_iter=10))
#         ])