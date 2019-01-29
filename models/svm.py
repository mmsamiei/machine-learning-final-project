import numpy
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

params = {'C': 1.0,
          'cache_size': 200,
          'class_weight': None,
          'coef0': 0.0,
          'decision_function_shape': 'ovr',
          'degree': 2,
          'gamma': 'scale',
          'kernel': 'linear',
          'max_iter': -1,
          'probability': False,
          'random_state': None,
          'shrinking': True,
          'tol': 0.001,
          'verbose': False}


def find_hyperparameter(X, y):
    clf = SVC()
    C = [1e-1, 1e0, 1e1]
    kernel = ['linear', 'poly', 'rbf']
    degree = [1, 2, 3]
    gamma = ['scale', 'auto']
    tol = [1e-3, 1e-2, 1e-1]
    random_grid = {
        'kernel': kernel,
        'degree': degree,
        'gamma': gamma,
        'tol': tol
    }
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=30, cv=3, n_jobs=-1)
    clf_random.fit(X, y)
    best_random_model = clf_random.best_estimator_
    scores = cross_val_score(best_random_model, X, y, cv=5)
    print("cross validation scores of best moddel are :", scores)
    print("mean of cross validation scores of best model is:", numpy.mean(scores))
    return best_random_model


def without_penalty(X, y):
    clf = SVC()
    clf.set_params(**params)
    clf.fit(X, y)
    return clf


def penalty_l2(X, y, l):
    clf = LinearSVC()
    clf.dual = False
    clf.max_iter = 2500
    clf.penalty = 'l2'
    clf.C = 1/l
    clf.fit(X, y)
    return clf

def penalty_l1(X, y, l):
    clf = LinearSVC()
    clf.dual = False
    clf.max_iter = 2500
    clf.penalty = 'l1'
    clf.C = 1/l
    clf.fit(X, y)
    return clf

if __name__ == '__main__':
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    X, y = shuffle(X, y)
    penalty_l1(X[:500], y[:500])
