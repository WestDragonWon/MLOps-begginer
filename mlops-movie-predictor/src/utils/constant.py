from enum import Enum

import torch.optim as optim

from model.movie_predictor import MoviePredictor


class CustomEnum(Enum):
    @classmethod
    def names(cls):
        return [member.name for member in list(cls)]

    @classmethod
    def validation(cls, name: str):
        names = [name.lower() for name in cls.names()]
        if name.lower() in names:
            return True
        else:
            raise ValueError(f"Invalid argument. Must be one of {cls.names()}")


class Models(CustomEnum):
    MOVIE_PREDICTOR = MoviePredictor


class Optimizers(CustomEnum):
    ADAM = optim.Adam
    RADAM = optim.RAdam
    NADAM = optim.NAdam
    SPARSEADAM = optim.SparseAdam
    SGD = optim.SGD
    RMSPROP = optim.RMSprop
    
    #생성