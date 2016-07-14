"""Keras neural network base model."""
import numpy as np
from ..window_model import FrameModel


class KerasBaseModel(FrameModel):
    """Convnet for frame prediction."""
    def __init__(self):
        self.model = None
        self.trained = False

    def _compile_model(self, input_shape, num_categories):
        """To be implemented by subclass."""
        pass

    def train(self, x, y):
        """Train."""
        self.model = self._compile_model(x[0].shape, y.shape[1])
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(
            x, y,
            callbacks=[early_stopping],
            verbose=1,
            validation_split=0.2,  # last 10% of data
            shuffle=True,
            nb_epoch=100,
        )
        self.trained = True

    def predict(self, x):
        """Predict."""
        assert(self.trained is True)
        return self.model.predict_classes(x)
