"""Model base class."""


class Model(object):
    def train(self, x_train, y_train):
        pass

    def predict(self, x):
        pass


class SequenceModel(Model)
