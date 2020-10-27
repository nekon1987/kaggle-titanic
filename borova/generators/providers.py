import abc

from pandas import DataFrame
from tensorflow.python.feature_column.dense_features import DenseFeatures
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.models import Model
from typing import List, Generator, Callable

from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from borova.data_models.data_frames import DataFrameWithFeatures
from borova.engine.data_variation import Variat
from borova.engine.patrick import Patrick


class ModelVariantsProvider(Patrick):
    @abc.abstractmethod
    def generate(self, V: Variat, input_features_layer: DenseFeatures) -> Generator[Model, None, None]:
        pass

class LossVariantsProvider(Patrick):
    @abc.abstractmethod
    def generate(self, V: Variat) -> Generator[Loss, None, None]:
        pass

class OptimizerVariantsProvider(Patrick):

    @abc.abstractmethod
    def generate(self, V: Variat) -> Generator[OptimizerV2, None, None]:
        pass


class InputDataProcessor(Patrick):

    # todo should be yielding this data
    @abc.abstractmethod
    def generate(self, V: Variat, input_data: DataFrame, is_training: bool) -> Generator[DataFrameWithFeatures, None, None]:
        pass