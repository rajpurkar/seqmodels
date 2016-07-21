from .. import model
from ..util import *

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class FrameModel(model.Model):
    def evaluate(self, x, y):
        debug('\n')
        debug(classification_report(
            y, self.predict(x)))


class CombinerModel(model.Model):
    def evaluate(self, x, y):
        pass


class WindowBasedModel(model.Model):
    def __init__(self, frame_model, combiner_model):
        self.frame_model = frame_model
        self.combiner_model = combiner_model
        self.output_length = None
        self.window_size = None
        self.stride = None
        
        self.left_epsilon=None
        self.right_epsilon=None
        self.only_positive=None
        self.X_TIME_COLUMN=None
        self.Y_TIME_COLUMN=None
        
        assert(isinstance(self.frame_model, FrameModel))
        assert(isinstance(self.combiner_model, CombinerModel))

    def train(self, x_train, y_train, evaluate_internals=True):
        def flatten(l):
            return np.array([x for sublist in l for x in sublist])

        self.output_length = len(y_train[0])
        windows_x, windows_y = self.get_windows(x_train, y_train)

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

    def get_windows_for_sequence(self, sequence, labels):
        """Get sliding window extractions and labels."""
        extractions = np.array([sequence[i: i + self.window_size, :] for i in \
            range(0, len(sequence) - self.window_size + 1, self.stride)])
        
        extraction_labels = np.zeros(len(extractions), dtype=np.int) - 1

        def get_non_zero_mask(left, right):
            mask = np.logical_and(labels_time - start_time > left,
                                  end_time - labels_time > right)
            label = np.flatnonzero(mask)
            return label

        for index, extraction in enumerate(extractions):
            start_time = extraction[0, self.X_TIME_COLUMN]
            end_time = extraction[-1, self.X_TIME_COLUMN]
            labels_time = labels[:, self.Y_TIME_COLUMN]
            label = get_non_zero_mask(self.left_epsilon, self.right_epsilon)
            if len(label) > 1:
                raise Warning("Overlapping labels. Reduce Epsilon boundaries")
            elif len(label) == 1:
                extraction_labels[index] = label[0]
            elif len(label) == 0:
                if not self.only_positive:
                    if len(get_non_zero_mask(-self.right_epsilon, -self.left_epsilon)) == 0:
                        extraction_labels[index] = 0
        
        extractions = extractions[extraction_labels > -1]
        extraction_labels = extraction_labels[extraction_labels > -1]

        return extractions, extraction_labels

    def get_windows(self, x_train, y_train):
        """Get sequence extraction pairs."""

        def roundMultiple(x, base=4):
            """Round n up to nearest multiple of base."""
            return int(base * round(float(x)/base))

        def auto_set_stride():
            self.stride = roundMultiple(
                int(self.window_size / 10), base=2)
            debug("Stride auto set to ", self.stride)

        def auto_set_window_size(sequence):
            threshold = (self.left_epsilon + self.right_epsilon) * 2
            time_arr = sequence[:, self.X_TIME_COLUMN]
            self.window_size = roundMultiple(
                np.argmax(time_arr > threshold), base=4)
            debug("Window size auto set to ", self.window_size)

        windows_x = []
        windows_y = []
        debug("Making windows...")
        if self.window_size is None:
            auto_set_window_size(x_train[0])
        if self.stride is None:
            auto_set_stride()

        for index in tqdm(range(len(x_train))):
            sequence_extractions, sequence_extraction_labels = \
                self.get_windows_for_sequence(
                    x_train[index], y_train[index])
            windows_x.append(sequence_extractions)
            windows_y.append(sequence_extraction_labels)
        return np.array(windows_x), np.array(windows_y)
