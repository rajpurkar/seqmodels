from models import model
from .load import *


class FrameModel(object):
    def train(self, x_train, y_train):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass


class FrameCombinerModel(object):
    def __init__(self):
        self.output_length = None

    def train(self, x_train, y_train):
        """Save output length."""
        self.output_length = len(y_train[0])

    def combine_outputs(self, y_s):
        test_mask = np.where(y_s != 0)[0]
        y_out = y_test[test_mask]
        return y_out[:output_length]

    def evaluate(self, x, y):
        pass


class WindowBasedModel(model.Model):
    def __init__(self, frame_model, frame_combiner_model):
        self.frame_model = frame_model
        self.frame_combiner_model = frame_combiner_model
        self.output_length = None
        assert(isinstance(self.frame_model, FrameModel))
        assert(isinstance(self.frame_combiner_model, FrameCombinerModel))

    def train(self, x_train, y_train, evaluate_internals=True):
        self.output_length = len(y_train[0])
        windows_x, windows_y = get_windows(x_train, y_train)
        if evaluate_internals:
            self._evaluate(x_train, y_train)

    def _evaluate(self, x_train, y_train):
        self.frame_model.evaluate(x_train, y_train)
        self.frame_combiner_model.evaluate(x_train, y_train)

    def predict(self, x):
        return np.ones((len(x), self.output_length))
