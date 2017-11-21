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
from featureEng import *

color = sns.color_palette()
#nltk.download('stopwords')
def run(name = 'NB1', n_comp = 50, ngram_cv = 3, ngram_tfidf= 5, eta = 0.1, gamma = 0):

    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    train_id = train_df['id'].values
    test_id = test_df['id'].values
    train_df, test_df = featureEng(train_df, test_df)
    author_mapping_dict = {'EAP':0, 'HPL':1 ,'MWS':2}
    train_y = train_df['author'].map(author_mapping_dict)

    def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.2, eta = 0.2, gamma = 0):
        param = {}
        param['objective'] = 'multi:softprob'
        param['max_depth'] = 5
        param['silent'] = True
        param['num_class'] = 3
        param['eval_metric'] = "mlogloss"
        param['min_child_weight'] = child
        param['subsample'] = 0.8
        param['colsample_bytree'] = colsample
        param['seed'] = seed_val
        param['learning_rate'] = eta
        param['n_estimators'] = 2000
        param['gamma'] = gamma
        param['nthread'] = 4
        param['scale_pos_weight'] = 1
        num_rounds = 5000

        plst = list(param.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
            xgtest = xgb.DMatrix(test_X, label=test_y)
            watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
            model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=False)
        else:
            xgtest = xgb.DMatrix(test_X)
            model = xgb.train(plst, xgtrain, num_rounds)

        pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
        if test_X2 is not None:
            xgtest2 = xgb.DMatrix(test_X2)
            pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
        return pred_test_y, pred_test_y2, model

    ###XGBoost model:
    cols_to_drop = ['id', 'text']
    train_X = train_df.drop(cols_to_drop+['author'], axis=1)
    test_X = test_df.drop(cols_to_drop, axis=1)

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_df.shape[0], 3])
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7,eta=eta, gamma=gamma)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    pred_full_test = pred_full_test / 5.
    print("XGB on new variables, cv scores : ", np.mean(cv_scores))


    out_df = pd.DataFrame(pred_full_test)
    out_df.columns = ['EAP', 'HPL', 'MWS']
    out_df.insert(0, 'id', test_id)
    out_df.to_csv(name+"_"+str(n_comp)+"_"+str(ngram_cv)+"_"+str(ngram_tfidf)+"_"+str(eta)+"_"+str(gamma)+"_submission.csv", index=False)

    ### Plot the important variables ###
    fig, ax = plt.subplots(figsize=(12,12))
    xgb.plot_importance(model, height=0.8, ax=ax)
    plt.show()

    cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y,axis=1))
    np.set_printoptions(precision=2)
    print(cnf_matrix)

    return np.mean(cv_scores)

    # Plot non-normalized confusion matrix
    #plt.figure(figsize=(8,8))
    #plt_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],title='Confusion matrix of XGB, without normalization')
    #plt.show()
