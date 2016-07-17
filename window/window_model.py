from .. import model
from ..util import *

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
        
        self.left_epsilon=0.15
        self.right_epsilon=0.15
        self.only_positive=False
        self.X_TIME_COLUMN=2
        self.Y_TIME_COLUMN=1
        
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

    def get_sequence_windows(self, sequence, labels):
        """Get sliding window extractions and labels."""
        def assign_window_extractions_labels(extractions, labels):
            """Assign window extraction labels."""
            extraction_labels = np.zeros(len(extractions), dtype=np.int)
            for index, extraction in enumerate(extractions):
                for label in labels:
                    start_time = extraction[0, self.X_TIME_COLUMN]
                    end_time = extraction[-1, self.X_TIME_COLUMN]
                    label_time = label[self.Y_TIME_COLUMN]
                    if(label_time - start_time > self.left_epsilon and
                       end_time - label_time > self.right_epsilon):
                            extraction_labels[index] = int(label[0])
            return extraction_labels

        def auto_set_window_size(sequence):
            threshold = (self.left_epsilon + self.right_epsilon) * 3 / 4
            time_arr = sequence[:, self.X_TIME_COLUMN]
            self.window_size = np.argmax(time_arr > threshold)

        def sliding_window_extractions(sequence):
            """Sliding window extraction."""
            length = len(sequence)
            extractions = []
            if self.window_size is None:
                auto_set_window_size(sequence)
            for i in range(length - self.window_size + 1):
                extraction = sequence[i: i + self.window_size, :]
                extractions.append(extraction)
            return np.array(extractions)

        extractions = sliding_window_extractions(sequence)
        extraction_labels = assign_window_extractions_labels(extractions, labels)
        if self.only_positive:
            extractions = extractions[extraction_labels > 0]
            extraction_labels = extraction_labels[extraction_labels > 0]
        return extractions, extraction_labels

    def get_windows(self, x_train, y_train):
        """Get sequence extraction pairs."""
        windows_x = []
        windows_y = []
        for index in range(len(x_train)):
            sequence_extractions, sequence_extraction_labels = \
                self.get_sequence_windows(
                    x_train[index], y_train[index])
            windows_x.append(sequence_extractions)
            windows_y.append(sequence_extraction_labels)
        return np.array(windows_x), np.array(windows_y)
