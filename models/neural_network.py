import numpy
from src.preprocessing import feature_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

def without_penalty():
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    X = feature_selection.dummy_selctor(X, 200)
    clf = MLPClassifier(max_iter=1000)
    hidden_layer_sizes = [int(x) for x in numpy.linspace(8, 128, 16)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    alpha = [0]
    learning_rate = ['constant', 'invscaling', 'adaptive']

    random_grid = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'alpha' : alpha,
        'learning_rate' : learning_rate
    }
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=20, cv=3, n_jobs=-1)
    clf_random.fit(X, y)
    best_random_model = clf_random.best_estimator_
    print(best_random_model.get_params())
    scores = cross_val_score(best_random_model, X, y, cv=5)
    print(scores)
    print(numpy.mean(scores))

if __name__ == '__main__':
    without_penalty()