from models import model
from . import load
from ..util import *

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class FrameModel(model.Model):
    def evaluate(self, x, y):
        debug('\n')
        debug(classification_report(
            y, self.predict(x), target_names=get_labels()))


class CombinerModel(model.Model):
    def evaluate(self, x, y):
        pass


class WindowBasedModel(model.Model):
    def __init__(self, frame_model, combiner_model):
        self.frame_model = frame_model
        self.combiner_model = combiner_model
        self.output_length = None
        assert(isinstance(self.frame_model, FrameModel))
        assert(isinstance(self.combiner_model, CombinerModel))

    def train(self, x_train, y_train, evaluate_internals=True):
        def flatten(l):
            return np.array([x for sublist in l for x in sublist])

        self.output_length = len(y_train[0])
        windows_x, windows_y = load.get_windows(x_train, y_train)

        debug('Window X shape: ', windows_x.shape)
        debug('Window Y shape: ', windows_y.shape)

        flat_x, flat_y = flatten(windows_x), flatten(windows_y)
        flat_y_one_hot = one_hot_encode_y(flat_y)

        debug('Frame X shape: ', flat_x.shape)
        debug('Frame Y shape: ', flat_y_one_hot.shape)
        debug('Y counts: ', dict(zip(*np.unique(flat_y, return_counts=True))))
        debug('\n')

        self.frame_model.train(flat_x, flat_y_one_hot)
        if (evaluate_internals):
            self.frame_model.evaluate(flat_x, flat_y)
        self.combiner_model.train(windows_y, y_train)

    def predict(self, x):
        return np.ones((len(x), 6))
