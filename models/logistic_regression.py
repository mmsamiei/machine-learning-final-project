from sklearn.linear_model import LogisticRegression
import numpy
from src.preprocessing import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import sknn

params = {'C': numpy.inf,
          'class_weight': None,
          'dual': False,
          'fit_intercept': True,
          'intercept_scaling': 1,
          'max_iter': 250,
          'multi_class': 'multinomial',
          'n_jobs': None,
          'penalty': 'l2',
          'random_state': None,
          'solver': 'newton-cg',
          'tol': 0.0001,
          'verbose': 0,
          'warm_start': False}


def find_hyperparameter(X, y):
    clf = LogisticRegression(C=numpy.inf, multi_class='multinomial', solver='newton-cg')
    scores = cross_val_score(clf, X, y, cv=5)
    print("cross validation scores of best moddel are :", scores)
    print("mean of cross validation scores of best model is:", numpy.mean(scores))
    return clf

def without_penalty(X, y):
    clf = LogisticRegression()
    clf.set_params(**params)
    clf.fit(X, y)
    return clf

def penalty_l1(X, y, l):
    clf = LogisticRegression().set_params(**params)
    clf.C = 1/l
    clf.penalty = 'l1'
    clf.solver = 'saga'
    scores = cross_val_score(clf, X, y, cv=5)
    print("cross validation scores of logistic regression model with l1 = {} are :".format(l), scores)
    print("mean of cross validation scores of logistic regression with l1 = {} model is:".format(l), numpy.mean(scores))
    clf.fit(X, y)
    return numpy.mean(scores), clf

def penalty_l2(X, y, l):
    clf = LogisticRegression().set_params(**params)
    clf.C = 1/l
    clf.penalty = 'l2'
    #scores = cross_val_score(clf, X, y, cv=5)
    #print("cross validation scores of logistic regression model with l2 = {} are :".format(l), scores)
    #print("mean of cross validation scores of logistic regression with l2 = {} model is:".format(l), numpy.mean(scores))
    clf.fit(X, y)
    return clf

if __name__ == '__main__':
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    without_penalty(X, y)
