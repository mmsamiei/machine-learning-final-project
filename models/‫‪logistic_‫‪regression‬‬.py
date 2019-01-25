from sklearn.linear_model import LogisticRegression
import numpy
from src.preprocessing import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import sknn

def without_penalty(X, y):
    X = feature_selection.dummy_selctor(X, 40)
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)
    clf = LogisticRegression(C=numpy.inf, multi_class='multinomial', solver='newton-cg')
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)


if __name__ == '__main__':
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    without_penalty(X, y)

