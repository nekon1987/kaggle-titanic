from itertools import tee

from borova.data_models.experiment import ExperimentComponentsAggregator
from borova.engine.data_variation import Variat
from borova.engine.patrick import Patrick
from borova.generators.providers import LossVariantsProvider, OptimizerVariantsProvider, ModelVariantsProvider, \
    InputDataProcessor


class LazyIteratorCell(object):
    Generator = None
    MaxYieldsPerIteration = int
    CurrentSubYield = int

class VariantGeneratorsAggregate(Patrick):
    Id = str
    Losses = LossVariantsProvider
    Optimizers = OptimizerVariantsProvider
    Models = ModelVariantsProvider
    InputData = InputDataProcessor

    lazy_iterator_memory = [] # TODO List[LazyIteratorCell]
    def lazy_forward_iterator(self, id, generator):

        iterator = next(filter(lambda x: x.Id == id,  self.lazy_iterator_memory), None)
        if iterator == None:
            #todo DRY with bellow code
            iterator = LazyIteratorCell()
            iterator.Id = id
            iterator.CurrentSubYield = 0
            iterators = tee(generator)
            iterator.Generator = iterators[0]
            iterator.MaxYieldsPerIteration = sum(1 for coolhack in iterators[1])
            self.lazy_iterator_memory.append(iterator)

        iterator.CurrentSubYield = iterator.CurrentSubYield + 1
        result = next(iterator.Generator, None)

        if result == None:
            iterator.CurrentSubYield  = 0
            # first remove original
            self.lazy_iterator_memory.remove(iterator)
            iterator = LazyIteratorCell()
            iterator.Id = id
            iterator.CurrentSubYield = 0
            iterators = tee(generator)
            iterator.Generator = iterators[0]
            iterator.MaxYieldsPerIteration = sum(1 for coolhack in iterators[1])
            self.lazy_iterator_memory.append(iterator)
            return next(iterator.Generator, None)

        return result


    def generate_variants_for_experiment(self, input_data_frame, is_training) -> ExperimentComponentsAggregator :

        variat = Variat()

        optimizer = self.lazy_forward_iterator('optimizer', self.Optimizers.generate(variat))
        loss = self.lazy_forward_iterator('loss', self.Losses.generate(variat))
        data = self.lazy_forward_iterator('data', self.InputData.generate(variat, input_data_frame, is_training))
        model = self.lazy_forward_iterator('model', self.Models.generate(variat, data.FeaturesLayer))

        return ExperimentComponentsAggregator(variat, optimizer, model, loss,  data.InputData)

    def calculate_number_of_combinations(self, input_data_frame, is_training):
        v = Variat()
        # todo: iterate over all properties to be safe in future
        # todo: calc on decorator variances + lists length
        #todo V should use dependency injection

        #todo this requires too much complexity and gives nothing - now I need to pass actual input dataframe?...

        def calculate_total_number_of_variants(generator):
            return sum(1 for coolhack in generator.generate(v)) * generator.generate.number_of_variations

        def calculate_total_number_of_variants_for_model_and_data():
            single_input_data = next(self.InputData.generate(v, input_data_frame, is_training))
            data_yields_per_iteration = sum(1 for coolhack in self.InputData.generate(v, input_data_frame, is_training))
            model_yields_per_iteration = sum(1 for coolhack in self.Models.generate(v, single_input_data.FeaturesLayer))
            models_variations_number = model_yields_per_iteration * self.Models.generate.number_of_variations
            input_data_variations_number =  data_yields_per_iteration * self.InputData.generate.number_of_variations
            return models_variations_number + input_data_variations_number

       # TODO Cannot calc models as they need actual data !! Should I provide it or refactor this all?
        total_variations_number =  calculate_total_number_of_variants(self.Optimizers) + \
                                   calculate_total_number_of_variants(self.Losses) + \
                                   calculate_total_number_of_variants_for_model_and_data()



        return total_variations_number
