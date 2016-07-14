"""Window."""
import numpy as np


def get_sequence_windows(
    sequence,
    labels,
    left_epsilon=150,
    right_epsilon=90,
    window_size=36,
    only_positive=False,
    X_TIME_COLUMN=3,
    Y_TIME_COLUMN=1
):
    """Get sliding window extractions and labels."""

    def assign_window_extractions_labels(extractions, labels):
        """Assign window extraction labels."""
        extraction_labels = np.zeros(len(extractions), dtype=np.int)
        for index, extraction in enumerate(extractions):
            for label in labels:
                start_time = extraction[0, X_TIME_COLUMN]
                end_time = extraction[-1, X_TIME_COLUMN]
                label_time = label[Y_TIME_COLUMN]
                if(label_time - start_time > left_epsilon and
                   end_time - label_time > right_epsilon):
                        extraction_labels[index] = int(label[0])
        return extraction_labels

    def sliding_window_extractions(sequence):
        """Sliding window extraction."""
        length = len(sequence)
        extractions = []
        for i in range(length - window_size + 1):
            extraction = sequence[i: i + window_size, :]
            extractions.append(extraction)
        return np.array(extractions)

    extractions = sliding_window_extractions(sequence)
    extraction_labels = assign_window_extractions_labels(extractions, labels)
    if only_positive:
        extractions = extractions[extraction_labels > 0]
        extraction_labels = extraction_labels[extraction_labels > 0]
    return extractions, extraction_labels


def get_windows(x_train, y_train, only_positive=False):
    """Get sequence extraction pairs."""
    windows_x = []
    windows_y = []
    for index in range(len(x_train)):
        sequence_extractions, sequence_extraction_labels = \
            get_sequence_windows(
                x_train[index], y_train[index], only_positive)
        windows_x.append(sequence_extractions)
        windows_y.append(sequence_extraction_labels)
    return np.array(windows_x), np.array(windows_y)
