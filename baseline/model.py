"""Simplest Model."""
from .. import model
import numpy as np


class Simple(model.SequenceModel):
    def __init__(self):
        self.output_length = None

    def train(self, x_train, y_train):
        """Save output length."""
        self.output_length = len(y_train[0])

    def predict(self, x):
        return np.ones((len(x), self.output_length))
