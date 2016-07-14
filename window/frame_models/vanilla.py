"""Vanilla neural network model."""
import numpy as np
from .. import window_model


class Vanilla(window_model.FrameModel):
    """Convnet for frame prediction."""
    def __init__(self):
        self.model = None
        self.trained = False

    def _compile_model(self, input_shape, num_categories):
        from keras.layers.core import Activation, Dense, Dropout, Reshape
        from keras.models import Sequential
        model = Sequential()
        model.add(
            Reshape((input_shape[0]*input_shape[1],), input_shape=input_shape))
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dense(num_categories))
        model.add(Activation('softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=["accuracy"]
        )
        return model

    def train(self, x, y):
        """Train."""
        self.model = self._compile_model(x[0].shape, y.shape[1])
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(
            x, y,
            callbacks=[early_stopping],
            verbose=2,
            validation_split=0.2,
            shuffle=True,
            nb_epoch=100,
        )
        self.trained = True

    def predict(self, x):
        """Predict."""
        assert(self.trained is True)
        return self.model.predict_classes(x)
