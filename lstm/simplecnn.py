import numpy
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Flatten, RNN, SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
from featureEng import *

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
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_id = train_df['id'].values
test_id = test_df['id'].values
#train_df, test_df = featureEng(train_df, test_df)
train_df = pd.read_csv("./train_df.csv")
test_df = pd.read_csv("./test_df.csv")

cols_to_drop =['id', 'text']
train_text = train_df.drop(cols_to_drop+['author'], axis=1).as_matrix()
train_text = np.expand_dims(train_text, axis=2)
test_text = test_df.drop(cols_to_drop, axis=1).as_matrix()
test_text = np.expand_dims(test_text, axis=2)
#rrain_text = train_df['text']
#test_text = test_df['text']

pca = PCA(n_components=200)
lca = TruncatedSVD(n_components=800, n_iter=2, random_state=42)

#train_text = lca.fit_transform(vectorizer.fit_transform(train_df.text.values))
train_author = LabelEncoder().fit_transform(train_df.author)

#train_text = vectorizer.fit_transform(trainall_df.text.values)
#X_sub = lca.fit_transform(vectorizer.fit_transform(testall_df.text.values))

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
#X_sub = ch2/transform(X_sub).toarray()

#tokenizer = Tokenizer(num_words=top_words)
#tokenizer.fit_on_texts(X_train)
#X_train = tokenizer.texts_to_sequences(X_train)
#X_test = tokenizer.texts_to_sequences(X_test)
#X#_sub = tokenizer.texts_to_sequences(test_text)

# truncate and pad input sequences
max_review_length = 50
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
#X_sub = sequence.pad_sequences(X_sub, maxlen=max_review_length)
X_sub = test_text
#We can now define, compile and fit our LSTM model.
#print(X_train[:4][:2],X_train.shape,X_test[:4][:2],y_train[:4][:2],y_test[:4][:2])

# create the model
embedding_vecor_length = 32
model = Sequential()
#model.add(Embedding(top_words, output_dim=embedding_vecor_length))
model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu',input_shape=X_train.shape[1:]))
model.add(AveragePooling1D(pool_size=2))
model.add(Conv1D(filters=4, kernel_size=12, padding='same', activation='sigmoid'))
#model.add(Conv1D(filters=5, kernel_size=20, padding='same', activation='relu'))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.1))
#model.add(LSTM(50))
#model.add(SimpleRNN(20))
#model.add(LSTM(200))
model.add(Dense(100))
model.add(Flatten())
#model.add(LSTM(50))
#model.add(Dense(10, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=32)
#Once fit, we estimate the performance of the model on unseen reviews.


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Loss: %.3f, Accuracy: %.3f%%" % (scores[0], scores[1]*100))
#For completeness, here is the full code listing for this LSTM network on the IMDB dataset.

y_pred = model.predict(X_test, verbose=1)
y_true = y_test
#print(y_pred[:4], y_true[:4])
log_loss = metrics.log_loss(y_true, y_pred)
cnf_matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
np.set_printoptions(precision=2)
print("Loss: %.3f" % (log_loss))
print(cnf_matrix)

##k=flod
#seed = 1
#kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)
#results = cross_val_score(model, X_test, y_test, cv = kfold)
#print("%2.f%(%.2f%%) " % (results.mean()*100, results.std()*100))

sub = model.predict(X_sub, verbose = 1)
predictions = pd.DataFrame(sub, columns=['EAP','HPL','MWS'])
predictions['id'] = test_df['id']
predictions.to_csv("cnn_submission.csv", index=False, columns=['id','EAP','HPL','MWS'])
