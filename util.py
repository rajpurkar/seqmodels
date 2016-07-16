from sklearn.preprocessing import OneHotEncoder

DEBUG = True


def debug(*arg):
    if DEBUG is True:
        print(*arg)


def one_hot_encode_y(y):
    """One hot encode."""
    enc = OneHotEncoder(sparse=False)
    return enc.fit_transform(y.reshape(-1, 1))
