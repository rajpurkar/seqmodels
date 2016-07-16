import seq2seq
from seq2seq.models import AttentionSeq2seq
from .. import model
from ..util import *

import numpy as np


class Sequence2SequenceModel(model.KerasModel):
    def __init__(self):
        super().__init__()
        self.output_length = 6
        self.MAX_LEN = 400
        self.HIDDEN_DIM = 10
        self.DEPTH = 4

    def _create_model(
        self,
        input_shape,
        num_categories
    ):
        model = AttentionSeq2seq(
            input_shape=(input_shape[0], input_shape[1]),
            hidden_dim=self.HIDDEN_DIM,
            output_length=self.output_length,
            output_dim=num_categories,
            depth=self.DEPTH
        )
        return model

    def train(self, x_train, y_train):
        from keras.preprocessing.sequence import pad_sequences
        x_train = pad_sequences(x_train, maxlen=self.MAX_LEN)

        y_train = np.array(y_train)[:, :, 0]
        self.output_length = y_train.shape[1]
        y_train = one_hot_encode_y(y_train).reshape(
            len(y_train), self.output_length, -1)

        super().train(x_train, y_train)
