import numpy
from src.preprocessing import feature_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier

def without_penalty(X, y):
    clf = AdaBoostClassifier()
    n_estimators = [int(x) for x in numpy.linspace(start=50, stop=200, num=10)]
    learning_rate = [0.8, 0.9, 1]
    random_grid = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate
    }
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=10, cv=3, n_jobs=-1)
    clf_random.fit(X, y)
    best_random_model = clf_random.best_estimator_
    scores = cross_val_score(best_random_model, X, y, cv=5)
    print(scores)
    print(numpy.mean(scores))
    best_random_model.fit(X, y)
    return best_random_model

if __name__ == '__main__':
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    without_penalty(X[:100], y[:100])