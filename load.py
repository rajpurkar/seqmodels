"""load module."""
import os
import joblib
import sklearn.cross_validation
from .util import *

class Loader(object):
    """Load a dataset created with the dataset collection pipeline.
    """
    def __init__(self, data_folder, processor, use_cached_if_available=True):
        self.x_train = self.x_test = self.y_train = self._y_test = None
        self.processor = processor
        self.RANDOM_SEED = 2
        self._load(data_folder, use_cached_if_available)

    def _load(self, data_folder, use_cached_if_available):
        """Run the pipeline to load the dataset.

        Returns the dataset with a train test split.
        """
        cached_filename = data_folder + '/cached'

        def check_cached_copy():
            return os.path.isfile(cached_filename)

        def load_cached():
            return joblib.load(cached_filename)

        def save_loaded(loaded):
            joblib.dump(loaded, cached_filename)

        if use_cached_if_available and check_cached_copy():
            debug("Using cached copy of dataset...")
            loaded = load_cached()
        else:
            debug("Loading dataset (not stored in cache)...")
            processed = self.processor.load_xy_pairs(data_folder)
            loaded = self._train_test(processed)
            debug("Saving to cache... (this may take some time)")
            save_loaded(loaded)


        (self.x_train, self.x_test, self.y_train, self._y_test) = loaded

    def _train_test(self, x_y_pairs):
        """Split the data pairs into a train test split."""
        x, y = zip(*x_y_pairs)
        x_train, x_test, y_train, y_test = sklearn.cross_validation.\
            train_test_split(
                x, y, test_size=0.2, random_state=self.RANDOM_SEED)
        return (x_train, x_test, y_train, y_test)

    def get_x_train(self):
        """Get the training data."""
        return self.x_train

    def get_x_test(self):
        """Get the test data."""
        return self.x_test

    def get_y_train(self):
        """Get the traning labels."""
        return self.y_train

    def get_y_test(self):
        """Get the test labels."""
        return self._y_test
