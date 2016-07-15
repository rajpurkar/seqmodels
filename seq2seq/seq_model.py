import seq2seq
from seq2seq.models import SimpleSeq2seq
from models import model
from ..util import *

import numpy as np


class Sequence2SequenceModel(model.KerasModel):
    def _create_model(self, input_shape, num_categories):
        model = SimpleSeq2seq(
            input_dim=4,
            hidden_dim=10,
            output_length=6,
            output_dim=5
        )
        return model

    def train(self, x_train, y_train):
        from keras.preprocessing.sequence import pad_sequences
        x_train = pad_sequences(x_train, maxlen=400)

        y_train = np.array(y_train)[:, :, 0]
        y_train = one_hot_encode_y(y_train).reshape(len(y_train), 6, -1)

        print(x_train.shape)
        print(y_train.shape)

        super().train(x_train, y_train)
