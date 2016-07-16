"""Vanilla neural network model."""
from ...model import KerasModel
from ..window_model import FrameModel


class Vanilla(KerasModel, FrameModel):
    """Simple 2 layer neural network using relu non-linearity."""
    def _create_model(self, input_shape, num_categories):
        from keras.layers.core import Activation, Dense, Dropout, Reshape
        from keras.models import Sequential
        model = Sequential()
        model.add(
            Reshape((input_shape[0]*input_shape[1],), input_shape=input_shape))
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dense(num_categories))
        model.add(Activation('softmax'))
        return model
