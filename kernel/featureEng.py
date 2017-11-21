import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import itertools
import xgboost as xgb
import re
import operator
from collections import Counter
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier
from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.metrics import confusion_matrix

from featureExt import *

#nltk.download('stopwords')
def featureEng(train_df, test_df, name = 'NB1', n_comp = 50, ngram_cv = 3, ngram_tfidf= 5):
    eng_stopwords = set(stopwords.words('english'))
    #pd.options.mode.chained_assignement = None

    cls = [(RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=10), "Random forest"),
        (LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3),"SVC-L1"),
        (LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3),"SVC-L2"),
        (SGDClassifier(alpha=.01, max_iter=50,penalty='l1'),'SGD-L1'),
        (SGDClassifier(alpha=.01, max_iter=50,penalty='l2'),'SGD-L2'),
        (SGDClassifier(alpha=.01, max_iter=50,penalty='elasticnet'),'SGD-ElasticNet'),
        (NearestCentroid(),'Nearest neighbor'),
        (MultinomialNB(alpha=.1),'NB1'),
        (BernoulliNB(alpha=.1),'NB2')]

    group_df = train_df.groupby('author')

    train_df['num_words'] = train_df['text'].apply(lambda x:len(str(x).split()))
    test_df['num_words'] = test_df['text'].apply(lambda x:len(str(x).split()))

    train_df['num_unique_words'] = train_df['text'].apply(lambda x:len(set(str(x).split())))
    test_df['num_unique_words'] = test_df['text'].apply(lambda x:len(set(str(x).split())))
    
    train_df['num_chars'] = train_df['text'].apply(lambda x:len(str(x)))
    test_df['num_chars'] = test_df['text'].apply(lambda x:len(str(x)))
    
 #   train_df['num_double_words'] = train_df['text'].apply(lambda x:len([w for w in list(x.split()) if len(w) > 1 and FreqDist(x.split())[w] == 2]))
  #  test_df['num_double_words'] = test_df['text'].apply(lambda x:len([w for w in list(x.split()) if len(w) > 1 and FreqDist(x.split())[w] == 2]))

    train_df['num_stopwords'] = train_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    test_df['num_stopwords'] = test_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    
    train_df['num_punctions'] = train_df['text'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))
    test_df['num_punctions'] = test_df['text'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))

    train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    test_df["num_words_upper"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    ## Number of title case words in the text ##
    train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    test_df["num_words_title"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    ## Average length of the words in the text ##
    train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    all_text_without_sw = ''
    for i in train_df.itertuples():
        all_text_without_sw = all_text_without_sw +  str(i.text)
    #getting counts of each words:
    counts = Counter(re.findall(r"[\w']+", all_text_without_sw))
    #deleting ' from counts
    del counts["'"]
    #getting top 50 used words:
    sorted_x = dict(sorted(counts.items(), key=operator.itemgetter(1),reverse=True)[:50])

    #Feature-5: The count of top used words.
    train_df['num_top'] = train_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in sorted_x]) )
    test_df['num_top'] = test_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in sorted_x]) )
    
	#Similarly lets identify the least used words:
    reverted_x = dict(sorted(counts.items(), key=operator.itemgetter(1))[:10000])
    #Feature-6: The count of least used words.
    train_df['num_least'] = train_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in reverted_x]) )
    test_df['num_least'] = test_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in reverted_x]) )

    train_df['unique_word_fraction'] = train_df['text'].apply(lambda row: unique_word_fraction(row))
    test_df['unique_word_fraction'] = test_df['text'].apply(lambda row: unique_word_fraction(row))

    train_df['stopwords_count'] = train_df['text'].apply(lambda row: stopwords_count(row))
    test_df['stopwords_count'] = test_df['text'].apply(lambda row: stopwords_count(row))

    train_df['punctuations_fraction'] = train_df['text'].apply(lambda row: punctuations_fraction(row))
    test_df['punctuations_fraction'] = test_df['text'].apply(lambda row: punctuations_fraction(row))

    train_df['char_count'] = train_df['text'].apply(lambda row: char_count(row))
    test_df['char_count'] = test_df['text'].apply(lambda row: char_count(row))

    train_df['fraction_noun'] = train_df['text'].apply(lambda row: fraction_noun(row))
    test_df['fraction_noun'] = test_df['text'].apply(lambda row: fraction_noun(row))

    train_df['fraction_adj'] = train_df['text'].apply(lambda row: fraction_adj(row))
    test_df['fraction_adj'] = test_df['text'].apply(lambda row: fraction_adj(row))

    train_df['fraction_verbs'] = train_df['text'].apply(lambda row: fraction_verbs(row))
    test_df['fraction_verbs'] = test_df['text'].apply(lambda row: fraction_verbs(row))

 #   train_df.loc[train_df['author']=='EAP'].shape[0]
    train_df['splited_text'] = train_df['text'].apply(lambda row: tokenixed_list(row))
    test_df['splited_text'] = test_df['text'].apply(lambda row: tokenixed_list(row))
    eap = train_df.loc[train_df['author']=='EAP']['splited_text'].values
    flat_eap = []
    eap_flat = [flat_eap.extend(x) for x in eap]
    flat_eap.__len__()
    eap_MM = make_higher_order_markov_model(3, flat_eap)

    hpl = train_df.loc[train_df['author']=='HPL']['splited_text'].values
    flat_hpl = []
    hpl_flat = [flat_hpl.extend(x) for x in hpl]
    flat_hpl.__len__()
    hpl_MM = make_higher_order_markov_model(3, flat_hpl)
    
    mws = train_df.loc[train_df['author']=='MWS']['splited_text'].values
    flat_mws = []
    mws_flat = [flat_mws.extend(x) for x in mws]
    flat_mws.__len__()
    mws_MM = make_higher_order_markov_model(3, flat_mws)

    # calculating markov chain based features
 #   train_df['EAP_markov_3'] = train_df['splited_text'].apply(lambda row: sent_to_prob_eap(sentence =row, mm1=eap_MM, p_eap=1/3, mm2=hpl_MM, mm3=mws_MM))
 #   test_df['EAP_markov_3'] = test_df['splited_text'].apply(lambda row: sent_to_prob_eap(sentence =row, mm1=eap_MM, p_eap=1/3, mm2=hpl_MM, mm3=mws_MM))
    
 #   train_df['HPL_markov_3'] = train_df['splited_text'].apply(lambda row: sent_to_prob_hpl(sentence = row, mm1=hpl_MM, p_hpl=1/3, mm2=eap_MM, mm3=mws_MM))
 #   test_df['HPL_markov_3'] = test_df['splited_text'].apply(lambda row: sent_to_prob_hpl(sentence = row, mm1=hpl_MM, p_hpl=1/3, mm2=eap_MM, mm3=mws_MM))

 #   train_df['MWS_markov_3'] = train_df['splited_text'].apply(lambda row: sent_to_prob_mws(sentence = row, mm1=mws_MM, p_mws=1/3, mm2=hpl_MM,mm3=eap_MM))
 #   test_df['MWS_markov_3'] = test_df['splited_text'].apply(lambda row: sent_to_prob_mws(sentence = row, mm1=mws_MM, p_mws=1/3, mm2=hpl_MM,mm3=eap_MM))
    #train_df[['MWS_markov_3', 'EAP_markov_3', 'HPL_markov_3', 'author']].head()
    del train_df['splited_text']
    del test_df['splited_text']
    #test_df[['MWS_markov_3', 'EAP_markov_3', 'HPL_markov_3']].head(50)

    author_mapping_dict = {'EAP':0, 'HPL':1 ,'MWS':2}
    train_y = train_df['author'].map(author_mapping_dict)
    ####compute the trauncated variables again ###
    #train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
    #test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))
    #train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    #test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


    cols_to_drop = ['id', 'text']
    train_X = train_df.drop(cols_to_drop+['author'], axis=1)
    test_X = test_df.drop(cols_to_drop, axis=1)
    
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,ngram_tfidf))
    full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

    def runClass(train_X, train_y, test_X, test_y, test_X2, name=name):
        model = [mod for mod in cls if mod[1] == name][0][0]
        model.fit(train_X, train_y)
        try:
            pred_test_y = model.predict_proba(test_X)
            pred_test_y2 = model.predict_proba(test_X2)
        except AttributeError:
            dec_f = model.decision_function(test_X)
            dec_f2 = model.decision_function(test_X2)
            pred_test_y = np.exp(dec_f) / np.sum(np.exp(dec_f))
            pred_test_y2 = np.exp(dec_f2) / np.sum(np.exp(dec_f2))
        return pred_test_y, pred_test_y2, model

    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_df.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runClass(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Word Tdidf MultiNB Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.

    # add the predictions as new features #
    train_df["nb_tfidf_eap"] = pred_train[:,0]
    train_df["nb_tfidf_hpl"] = pred_train[:,1]
    train_df["nb_tfidf_mws"] = pred_train[:,2]
    test_df["nb_tfidf_eap"] = pred_full_test[:,0]
    test_df["nb_tfidf_hpl"] = pred_full_test[:,1]
    test_df["nb_tfidf_mws"] = pred_full_test[:,2]


    ###SVD on word TFIDF
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='randomized')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

    train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

    ### Fit transform the count vectorizer ###
    tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,ngram_cv))
    tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_df.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runClass(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Word CV MultiNB Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.

    # add the predictions as new features #
    train_df["nb_cvec_eap"] = pred_train[:,0]
    train_df["nb_cvec_hpl"] = pred_train[:,1]
    train_df["nb_cvec_mws"] = pred_train[:,2]
    test_df["nb_cvec_eap"] = pred_full_test[:,0]
    test_df["nb_cvec_hpl"] = pred_full_test[:,1]
    test_df["nb_cvec_mws"] = pred_full_test[:,2]

    ### Fit transform the tfidf vectorizer ###
    tfidf_vec = CountVectorizer(ngram_range=(1,ngram_cv), analyzer='char')
    tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_df.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runClass(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Char CV MultiNB Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.

    # add the predictions as new features #
    train_df["nb_cvec_char_eap"] = pred_train[:,0]
    train_df["nb_cvec_char_hpl"] = pred_train[:,1]
    train_df["nb_cvec_char_mws"] = pred_train[:,2]
    test_df["nb_cvec_char_eap"] = pred_full_test[:,0]
    test_df["nb_cvec_char_hpl"] = pred_full_test[:,1]
    test_df["nb_cvec_char_mws"] = pred_full_test[:,2]

    ### Fit transform the tfidf vectorizer ###
    tfidf_vec = TfidfVectorizer(ngram_range=(1,ngram_tfidf), analyzer='char')
    full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())

    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_df.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runClass(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Char Tfidf MultiNB Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.

    # add the predictions as new features #
    train_df["nb_tfidf_char_eap"] = pred_train[:,0]
    train_df["nb_tfidf_char_hpl"] = pred_train[:,1]
    train_df["nb_tfidf_char_mws"] = pred_train[:,2]
    test_df["nb_tfidf_char_eap"] = pred_full_test[:,0]
    test_df["nb_tfidf_char_hpl"] = pred_full_test[:,1]
    test_df["nb_tfidf_char_mws"] = pred_full_test[:,2]

    ###SVD on Character TFIDF
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='randomized')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
        
    train_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
    
    return train_df, test_df
