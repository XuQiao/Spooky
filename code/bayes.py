
# coding: utf-8

# ## Simple Bayes classifier

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
import sklearn.metrics as mts

trainall_df = pd.read_csv("../input/train.csv")
train_df_X, test_df_X, train_df_Y, test_df_Y = train_test_split(trainall_df.text, trainall_df.author, test_size = 0.2, random_state = 1)
#test_df = pd.read_csv("../input/test.csv")

unigrams_pipeline = Pipeline([('text', CountVectorizer()),('mnb', MultinomialNB())])
unigrams_pipeline.fit(train_df_X, train_df_Y)
#test_df_Y = LabelBinarizer().fit_transform(test_df_Y)

predictions = unigrams_pipeline.predict_proba(test_df_X)
score = mts.log_loss(test_df_Y, predictions)
#, labels=['EAP','HPL','MWS'], pos_label=0, average='weighted', sample_weight=None)
print('log_loss = ', score)
#predictions = pd.DataFrame(
#    unigrams_pipeline.predict_proba(test_df.text),
#    columns=unigrams_pipeline.classes_
#                           )
#predictions['id'] = test_df['id']
#predictions.to_csv("submission.csv", index=False, columns=['id','EAP','HPL','MWS'])

