import numpy
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
n_gram = 2
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
#vectorizer = CountVectorizer(ngram_range=(1,n_gram), token_pattern = r'\b\w+\b', min_df = 1)
ch2 = SelectKBest(chi2, k='all')

# read in data
trainall_df = pd.read_csv("../input/train.csv")
testall_df = pd.read_csv("../input/test.csv")

pca = PCA(n_components=200)
lca = TruncatedSVD(n_components=800, n_iter=2, random_state=42)


train_text = lca.fit_transform(vectorizer.fit_transform(trainall_df.text.values))
train_author = LabelEncoder().fit_transform(trainall_df.author)

#train_text = vectorizer.fit_transform(trainall_df.text.values)
X_sub = lca.fit_transform(vectorizer.fit_transform(testall_df.text.values))

#X_sub = vectorizer.transform(testall_df.text.values)
#X_sub = ch2.transform(X_sub).toarray()

train_author = to_categorical(train_author)
# split a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(train_text, train_author, test_size = 0.2, random_state = 1)

#X_train = vectorizer.fit_transform(X_train)
#X_test = vectorizer.transform(X_test)
#X_sub = vectorizer.transform(X_sub)
# y_train = LabelBinarizer().fit_transform(y_train)
# y_test = LabelBinarizer().fit_transform(y_test)

#X_train = ch2.fit_transform(X_train,y_train).toarray()
#X_test = ch2.transform(X_test).toarray()
#X_sub = ch2.transform(X_sub).toarray()

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
X_sub = sequence.pad_sequences(X_sub, maxlen=max_review_length)
#We can now define, compile and fit our LSTM model.
print(X_train[:4],X_train.shape,X_test[:4],X_test.shape)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length)) #, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=1, batch_size=64)
#Once fit, we estimate the performance of the model on unseen reviews.


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Loss: %.3f, Accuracy: %.3f%%" % (scores[0], scores[1]*100))
#For completeness, here is the full code listing for this LSTM network on the IMDB dataset.

y_pred = model.predict(X_test, verbose=1)
y_true = y_test
print(y_pred[:4], y_true[:4])
log_loss = metrics.log_loss(y_true, y_pred)
print("Loss: %.3f" % (log_loss))

##k=flod
#seed = 1
#kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)
#results = cross_val_score(model, X_test, y_test, cv = kfold)
#print("%2.f%(%.2f%%) " % (results.mean()*100, results.std()*100))

sub = model.predict(X_sub, verbose = 1)
predictions = pd.DataFrame(sub, columns=['EAP','HPL','MWS'])
predictions['id'] = testall_df['id']
predictions.to_csv("submission.csv", index=False, columns=['id','EAP','HPL','MWS'])
