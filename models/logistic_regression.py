from sklearn.linear_model import LogisticRegression
import numpy
from src.preprocessing import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import sknn

def find_hyperparameter(X, y):
    clf = LogisticRegression(C=numpy.inf, multi_class='multinomial', solver='newton-cg')
    scores = cross_val_score(clf, X, y, cv=5)
    print("cross validation scores of best moddel are :", scores)
    print("mean of cross validation scores of best model is:", numpy.mean(scores))
    return clf


if __name__ == '__main__':
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    without_penalty(X, y)

