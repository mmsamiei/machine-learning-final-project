def dummy_selctor(X, l):
    if( l == 'all'):
        return X
    data = X.T[0:l].T
    return data
