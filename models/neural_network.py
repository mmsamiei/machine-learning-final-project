import numpy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

params = {'activation': 'tanh',
          'alpha': 0,
          'batch_size': 'auto',
          'beta_1': 0.9,
          'beta_2': 0.999,
          'early_stopping': False,
          'epsilon': 1e-08,
          'hidden_layer_sizes': 56,
          'learning_rate': 'constant',
          'learning_rate_init': 0.001,
          'max_iter': 1000,
          'momentum': 0.9,
          'n_iter_no_change': 10,
          'nesterovs_momentum': True,
          'power_t': 0.5,
          'random_state': None,
          'shuffle': True,
          'solver': 'adam',
          'tol': 0.0001,
          'validation_fraction': 0.1,
          'verbose': False,
          'warm_start': False}


def find_hyperparameter(X, y):
    clf = MLPClassifier(max_iter=1000)
    hidden_layer_sizes = [int(x) for x in numpy.linspace(8, 128, 16)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    alpha = [0]
    # learning_rate = ['constant', 'invscaling', 'adaptive']
    random_grid = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'alpha': alpha,
        # 'learning_rate': learning_rate
    }
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=32, cv=3, n_jobs=-1)
    clf_random.fit(X, y)
    best_random_model = clf_random.best_estimator_
    scores = cross_val_score(best_random_model, X, y, cv=5)
    print("cross validation scores of best moddel are :", scores)
    print("mean of cross validation scores of best model is:", numpy.mean(scores))
    return best_random_model


def without_penalty(X, y):
    # X = feature_selection.dummy_selctor(X, 200)
    clf = MLPClassifier(max_iter=1000)
    hidden_layer_sizes = [int(x) for x in numpy.linspace(8, 128, 16)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    alpha = [0]
    learning_rate = ['constant', 'invscaling', 'adaptive']

    random_grid = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'alpha': alpha,
        'learning_rate': learning_rate
    }
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=32, cv=3, n_jobs=-1)
    clf_random.fit(X, y)
    best_random_model = clf_random.best_estimator_
    print(best_random_model.get_params())
    scores = cross_val_score(best_random_model, X, y, cv=5)
    print(scores)
    print(numpy.mean(scores))
    return best_random_model

def penalty_l2(X, y, l):
    clf = MLPClassifier().set_params(**params)
    clf.alpha = l
    clf.penalty = 'l2'
    clf.fit(X, y)
    return clf

if __name__ == '__main__':
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    without_penalty(X, y)
