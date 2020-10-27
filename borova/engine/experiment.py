import pandas as pd
from pandas import DataFrame
from tensorflow.python.keras.models import Model

from borova.engine.patrick import Patrick
from borova.engine.session import BorovaSession
from borova.generators.providers import InputDataProcessor, ModelVariantsProvider, OptimizerVariantsProvider, \
    LossVariantsProvider
from borova.generators.variants_generation import VariantGeneratorsAggregate


class BorovaExperiment(Patrick):

    best_model = Model
    input_data = DataFrame

    def setup(self, input_data_processor: InputDataProcessor, models_provider: ModelVariantsProvider,
              optimizers_provider: OptimizerVariantsProvider, losses_provider: LossVariantsProvider):
        self.models_provider = models_provider
        self.optimizers_provider = optimizers_provider
        self.losses_provider = losses_provider
        self.input_data_processor = input_data_processor

    def load_data(self, file_name: str):
        self.input_data = pd.read_csv(file_name)

    def __prepare_variations_generators(self) -> VariantGeneratorsAggregate:
        variant_generators = VariantGeneratorsAggregate()
        variant_generators.Losses = self.losses_provider
        variant_generators.Optimizers = self.optimizers_provider
        variant_generators.InputData = self.input_data_processor
        variant_generators.Models = self.models_provider
        return variant_generators

    def run(self, session_name: str):
        print('Starking like there is no tomorrow')
        variant_generators_aggregate = self.__prepare_variations_generators()
        session = BorovaSession(variant_generators_aggregate, session_name)
        session.run(self.input_data)

    # todo: include concept of leaderboard - this may need to be reworked to accompany it
    def validate_model(self, model_package_path: str, input_data_file: str, session_name: str):
        input_data = pd.read_csv(input_data_file)
        variant_generators_aggregate = self.__prepare_variations_generators()
        session = BorovaSession(variant_generators_aggregate, session_name)
        return session.validate(model_package_path, input_data)


