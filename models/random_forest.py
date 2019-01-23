import numpy
from src.preprocessing import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def without_penalty():
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    X = feature_selection.dummy_selctor(X, 1)
    clf = RandomForestClassifier()
    n_estimators = [int(x) for x in numpy.linspace(start=50, stop=300, num=10)]
    max_depth = [int(x) for x in numpy.linspace(5, 200, num=10)]
    min_samples_split = [3, 5, 10]
    min_samples_leaf = [1, 3, 5]
    random_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }
    size_of_search_space = len(n_estimators) * len(max_depth) * len(min_samples_split) * len(min_samples_leaf)
    n_iter = size_of_search_space * 0.1
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=n_iter, cv=3, n_jobs=-1)
    clf_random.fit(X, y)
    # wait :)
    brp = best_random_parameters = clf_random.best_params_
    n_estimators = [int(x) for x in numpy.linspace(start=brp['n_estimators']-50, stop=brp['n_estimators']+50, num=5)]
    n_estimators = [x for x in n_estimators if x > 0]
    max_depth = [int(x) for x in numpy.linspace(brp['max_depth']-20, brp['max_depth']+20, num=5)]
    max_depth = [x for x in max_depth if x > 0]
    if(brp['min_samples_leaf'] == 1):
        min_samples_leaf = [1 , 3 ]
    elif(brp['min_samples_leaf'] == 3):
        min_samples_leaf = [1, 3, 5]
    else:
        min_samples_leaf = [3, 5, 7]
    complete_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=complete_grid,
                               cv=3, n_jobs=-1)
    grid_search.fit(X, y)
    best_grid = grid_search.best_estimator_
    scores = cross_val_score(best_grid, X, y, cv=5)
    print("cv scores of the best random forest is:")
    print(scores)
    print('mean scores of cv scores are:  {:0.3f}%.'.format(100 * numpy.mean(scores)))

if __name__ == '__main__':
    without_penalty()