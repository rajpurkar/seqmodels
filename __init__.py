from .model import *
from .window.window_model import CombinerModel, WindowBasedModel
from .window.frame_models.vanilla import Vanilla


class BestModel(WindowBasedModel):
    def __init__(self):
        frame_model = Vanilla()
        combiner_model = CombinerModel()
        super().__init__(frame_model, combiner_model)
