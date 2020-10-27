import utility.util as ut
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

ut.widen_df_display()

dialogue = ut.read_csv(r'D:\Ravi\Lab\Chatbot_Training\archive\Ubuntu-dialogue-corpus\dialogueText_196.csv')

print(dialogue.head())

ut.to_db(dialogue, 'dialogueText_196.db', 'Dialogue')