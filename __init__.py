from .window import *
from .model import *
from .baseline import *


class BestModel(WindowBasedModel):
    def __init__(self):
        frame_model = FrameModel()
        combiner_model = CombinerModel()
        super().__init__(frame_model, combiner_model)
