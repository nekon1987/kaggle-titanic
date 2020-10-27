import os, yaml
from datetime import datetime
import tensorflow as tf

from pandas import DataFrame
from tensorflow.python.keras.models import Model

from borova.data_models.experiment import ExperimentResult
from borova.engine.data_variation import Variat
from borova.generators.variants_generation import VariantGeneratorsAggregate
from borova.helpers.tensorflow import DataFrameHelper

import matplotlib.pyplot as plt

class BorovaSession(object):

    best_result = ExperimentResult
    session_name = str
    variant_generators_aggregate = VariantGeneratorsAggregate

    def __init__(self, variant_generators_aggregate: VariantGeneratorsAggregate, session_name: str):
        self.variant_generators_aggregate = variant_generators_aggregate
        self.best_result = ExperimentResult()
        self.session_name = session_name


    def run(self, input_data: DataFrame):
        iterations_left = self.variant_generators_aggregate.calculate_number_of_combinations(is_training=True,
                                                                                             input_data_frame=input_data)
        loss_scores = []
        while iterations_left > 0:
            # todo: session should be aware and hold state of current key providers and all the variables they
            # todo  randomized so we can save them if they perform well - bellow should peehaps return tupple
            aggregator = self.variant_generators_aggregate\
                .generate_variants_for_experiment(is_training=True, input_data_frame=input_data)

            aggregator.Model.compile(optimizer=aggregator.Optimizer, loss=aggregator.Loss, metrics=['accuracy'])

            train_split = aggregator.Variat.R('split', 0.6, 0.95, True)
            test_split = 1 - train_split
            epochs_number = int(aggregator.Variat.R('epochs', 150, 700, True))

            train_ds, validation_ds = DataFrameHelper.get_test_and_validation(aggregator.InputData, train_size=train_split, test_size=test_split)
            history = aggregator.Model.fit(train_ds, validation_data=validation_ds, epochs=epochs_number, verbose=0) # todo move epoch outside
            # loss_score = 2 - (history.history['loss'][-1] + history.history['val_loss'][-1])
            loss_score = (history.history['accuracy'][-1] + history.history['val_accuracy'][-1])/ 2

            loss_scores.append(loss_score)
            plt.plot(loss_scores)
            plt.show()

            iterations_left = iterations_left - 4 # todo  be calculated as number of components in aggregate
            print('Remaining itterations:' + str(iterations_left))


            if loss_score > self.best_result.LossScore:
                self.best_result = ExperimentResult.from_aggregator(aggregator, loss_score)
                self.save_model(self.best_result.Model, self.best_result.Variat, self.best_result.LossScore, self.session_name)

            self.best_result.Variat.print_traced_values('### Best Model @ ' + str(self.best_result.LossScore))
            aggregator.Variat.print_traced_values('### Currently Tested @ ' + str(loss_score))

        print('session running')

    def save_model(self, model: Model, variat: Variat, score: float, description: str) -> str:
        test_descriptive_label = datetime.now().strftime("%Y%m%d%H%M%S") + '-' + description + '-' + str(score)
        model_directory = os.getcwd() + '\\' + 'network_models\\borova\\' + test_descriptive_label
        os.mkdir(model_directory)
        model.save(filepath=model_directory, save_format='tf')
        with open(model_directory + '\\variat.json', 'w') as f:
            yaml.dump(variat, f)
        return model_directory

    def validate(self, package_directory: str, input_data: DataFrame):
        (model, variat) = self.__load_model__(package_directory)
        input_processed = next(self.variant_generators_aggregate.InputData.generate(variat, input_data, is_training=False))
        test_ds = tf.data.Dataset.from_tensor_slices(dict(input_processed.InputData))
        test_ds = test_ds.batch(892)
        return model.predict(x=test_ds)

    def __load_model__(self, package_directory: str) -> (Model, Variat):
        variat_path  = package_directory + '\\variat.json'
        model_path = package_directory
        model = tf.keras.models.load_model(model_path)
        #model = tf.saved_model.load(model_path)
        with open(variat_path, 'r') as f:
            variat = yaml.load(f)
        return model, variat


