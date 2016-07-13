from .window import *
from .model import *
from .baseline import *


class BestModel(WindowBasedModel):
    def __init__(self):
        frame_model = FrameModel()
        frame_combiner_model = FrameCombinerModel()
        super().__init__(frame_model, frame_combiner_model)
