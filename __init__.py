from .model import *
from .window.window_model import CombinerModel, WindowBasedModel
from .window.frame_models.recurrent import RecurrentModel
from .seq2seq.seq_model import Sequence2SequenceModel


class BestModel(Sequence2SequenceModel):
    pass


"""
class BestModel(WindowBasedModel):
    def __init__(self):
        frame_model = RecurrentModel()
        combiner_model = CombinerModel()
        super().__init__(frame_model, combiner_model)
"""