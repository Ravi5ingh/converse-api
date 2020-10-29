import utility.util as ut
import models.training.ensemble_word2vec_classifier as ec
import pandas as pd
import numpy as np
import sklearn.neural_network as nn

ut.widen_df_display()

classifier = ut.read_pkl('models/model.pkl')

pred = classifier.predict('i think someone has done fraud with me')

print(pred)