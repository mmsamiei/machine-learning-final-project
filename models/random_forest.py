import numpy
from src.preprocessing import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def without_penalty():
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    X = feature_selection.dummy_selctor(X, 40)
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier(n_estimators = 100 , max_depth=4, min_samples_leaf=1)
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)


if __name__ == '__main__':
    without_penalty()
