import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    X = numpy.loadtxt("../data/Train/X_train.txt")
    y = numpy.loadtxt("../data/Train/y_train.txt")
    X_test = numpy.loadtxt("../data/Test/X_test.txt")
    y_test = numpy.loadtxt("../data/Test/y_test.txt")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)
    lda = LDA(n_components=10)
    X_train_new = lda.fit_transform(X_train, y_train)
    clf = MLPClassifier(hidden_layer_sizes=(54), activation='tanh')
    clf.fit(X_train_new, y_train)
    validation_Score = clf.score(lda.transform(X_valid), y_valid)
    print("validation score: {}".format(validation_Score))
    # now we fit model on total of training data and test on test data
    X_train = X
    y_train = y
    X_train_new = lda.fit_transform(X_train, y_train)
    print("shape of X_train is :", X_train_new.shape)
    clf.fit(X_train_new, y_train)
    X_test_new = lda.transform(X_test)
    y_pred = clf.predict(X_test_new)
    test_score = accuracy_score(y_test, y_pred)
    print("test score is: {}".format(test_score))

