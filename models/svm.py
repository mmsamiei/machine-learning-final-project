import numpy
from sklearn.svm import SVC
from src.preprocessing import feature_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


def without_penalty():
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    X = feature_selection.dummy_selctor(X, 20)
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
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=10, cv=3, n_jobs=-1)
    clf_random.fit(X, y)
    best_random_model = clf_random.best_estimator_
    scores = cross_val_score(best_random_model, X, y, cv=5)
    print(scores)
    print(numpy.mean(scores))


if __name__ == '__main__':
    without_penalty()
