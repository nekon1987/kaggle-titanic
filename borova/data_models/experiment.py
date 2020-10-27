from pandas import DataFrame
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from borova.engine.data_variation import Variat
from borova.generators.providers import InputDataProcessor

class ExperimentComponentsAggregator(object):

    Variat = Variat
    Optimizer =  OptimizerV2
    Model = Model
    Loss = Loss
    InputData = DataFrame

    def __init__(self, variat: Variat, optimizer: OptimizerV2, model: Model, loss: Loss, inputData: DataFrame):
        self.Variat = variat
        self.Optimizer = optimizer
        self.Model = model
        self.Loss = loss
        self.InputData = inputData


class ExperimentResult(ExperimentComponentsAggregator):

    LossScore = float # todo - make into dict of metrics

    def __init__(self, loss_score: float = 0):
        self.LossScore = loss_score

    @staticmethod
    def from_aggregator(aggregator: ExperimentComponentsAggregator, loss_score: float):
        result = ExperimentResult(loss_score)
        result.__dict__.update(aggregator.__dict__)
        return result

    # def __init__(self, aggregator: ExperimentComponentsAggregator, loss_score: float):
    #     self.Variat = aggregator.Variat
    #     self.InputDataProcessor = aggregator.
    #     self.Model = model,
    #     self.LossScore = loss_score