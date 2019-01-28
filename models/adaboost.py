import numpy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# params = {'algorithm': 'SAMME.R',
#           'base_estimator': None,
#           'learning_rate': 0.8,
#           'n_estimators': 83,
#           'random_state': None}

params = {'algorithm': 'SAMME.R',
          'base_estimator__class_weight': None,
          'base_estimator__criterion': 'gini',
          'base_estimator__max_depth': 4,
          'base_estimator__max_features': None,
          'base_estimator__max_leaf_nodes': None,
          'base_estimator__min_impurity_decrease': 0.0,
          'base_estimator__min_impurity_split': None,
          'base_estimator__min_samples_leaf': 1,
          'base_estimator__min_samples_split': 2,
          'base_estimator__min_weight_fraction_leaf': 0.0,
          'base_estimator__presort': False,
          'base_estimator__random_state': None,
          'base_estimator__splitter': 'best',
          'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,
                                                   max_features=None, max_leaf_nodes=None,
                                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                                   min_samples_leaf=1, min_samples_split=2,
                                                   min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                                   splitter='best'),
          'learning_rate': 0.6,
          'n_estimators': 104,
          'random_state': None}


def find_hyperparameter(X, y):
    base_estimator = DecisionTreeClassifier(max_depth=4)
    clf = AdaBoostClassifier(n_estimators=100, base_estimator=base_estimator)
    n_estimators = [int(x) for x in numpy.linspace(start=50, stop=120, num=10)]
    learning_rate = [0.6, 0.8, 1]
    random_grid = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate
    }
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=10, cv=3, n_jobs=-1)
    clf_random.fit(X, y)
    best_random_model = clf_random.best_estimator_
    scores = cross_val_score(best_random_model, X, y, cv=5)
    print("cross validation scores of best moddel are :", scores)
    print("mean of cross validation scores of best model is:", numpy.mean(scores))
    return best_random_model


def without_penalty(X, y):
    clf = AdaBoostClassifier()
    clf.set_params(**params)
    clf.fit(X, y)
    return clf


if __name__ == '__main__':
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    find_hyperparameter(X, y)
