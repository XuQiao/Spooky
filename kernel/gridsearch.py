from sklearn.model_selection import ParameterGrid
from runxgboost import *

params = {'name':['SGD-L1'], 
         'n_comp':[50],
         'ngram_cv':[2],
         'ngram_tfidf':[5],
         'eta':[0.05],
         'gamma':[0.1]
         }

logloss = []

for param in ParameterGrid(params):
    name = param['name']
    n_comp = param['n_comp']
    ngram_cv = param['ngram_cv']
    ngram_tfidf = param['ngram_tfidf']
    eta = param['eta']
    gamma = param['gamma']
    logloss.append((name, n_comp, ngram_cv, ngram_tfidf, run(name = name, n_comp = n_comp, ngram_cv = ngram_cv, ngram_tfidf = ngram_tfidf, eta = eta, gamma = gamma)))

print(logloss)
