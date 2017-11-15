from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets

import spacy
from spacy.util import minibatch, compounding

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd

trainall_df = pd.read_csv("../input/train.csv")
testall_df = pd.read_csv("../input/test.csv")
X_train, X_test, y_train, y_test = train_test_split(trainall_df.text, trainall_df.author, test_size = 0.2, random_state = 1)
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

output_dir = './model'
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
doc2 = [nlp2(test_text).cats for test_text in trainall_df.text]
y_pred = [list(doc.values()) for doc in doc2]
log_loss = metrics.log_loss(LabelBinarizer().fit_transform(trainall_df.author), y_pred)
print("log_loss:   %0.3f" % log_loss)

sub = [nlp2(test_text).cats for test_text in testall_df.text]
y_sub = [list(doc.values()) for doc in sub]
predictions = pd.DataFrame(y_sub, columns=['EAP','HPL','MWS'])
predictions['id'] = testall_df['id']
predictions.to_csv("submission.csv", index=False, columns=['id','EAP','HPL','MWS'])
