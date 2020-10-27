import math
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python import feature_column
from tensorflow.python.feature_column.dense_features import DenseFeatures
from tensorflow.python.keras.models import Model
from pandas import DataFrame
from typing import Generator
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow as tf

from borova.attributes.variated import Variated
from borova.data_models.data_frames import DataFrameWithFeatures
from borova.engine.data_variation import Variat
from borova.engine.experiment import BorovaExperiment
from borova.generators.providers import ModelVariantsProvider, OptimizerVariantsProvider, LossVariantsProvider, \
    InputDataProcessor
from borova.helpers.tensorflow import DataFrameHelper
from helpers.KaggleHelper import KaggleHelper

class Mark1_ModelsProvider(ModelVariantsProvider):
    @Variated(number_of_variations=1000)
    def generate(self, V: Variat, input_features_layer: DenseFeatures) -> Generator[Model, None, None]:

        model = tf.keras.Sequential([
            input_features_layer,
            layers.Dense(158, activation='selu', kernel_initializer=initializers.lecun_normal()),
            layers.Dropout(0.2),
            layers.Dense(168, activation='swish', kernel_initializer=initializers.GlorotNormal()),
            layers.Dropout(0.2),
            layers.Dense(178, activation='swish', kernel_initializer=initializers.GlorotNormal()),
            layers.Dropout(0.2),
            layers.Dense(188, activation='selu', kernel_initializer=initializers.lecun_normal()),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid", kernel_initializer=initializers.lecun_normal())
        ])

        yield model

class Mark1_OptimizersProvider(OptimizerVariantsProvider):
    @Variated(number_of_variations=50)
    def generate(self, V: Variat):
        yield tf.keras.optimizers.Adam(learning_rate=V.R("lr1", 0.001, 0.01), beta_1=0.9,
                                       beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam1')
        yield tf.keras.optimizers.Adam(learning_rate=V.R("lr2", 0.001, 0.01), beta_1=V.R("lr3", 0.7, 0.9),
                                       beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam2')
        yield tf.keras.optimizers.Adam(learning_rate=V.R("lr4", 0.001, 0.01), beta_1=V.R("lr5", 0.7, 0.9),
                                       beta_2=V.R("lr6", 0.6, 0.999), epsilon=1e-07, amsgrad=False, name='Adam2')

class Mark1_LossProvider(LossVariantsProvider):
    @Variated(number_of_variations=50)
    def generate(self, V: Variat):
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        yield binary_crossentropy
        binary_crossentropy_with_smothing = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=V.R("bc1", 0.1, 1))
        yield binary_crossentropy_with_smothing
        binary_crossentropy_no_logits = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        yield binary_crossentropy_no_logits

class Mark1_InputDataProcessor(InputDataProcessor):

    scaler = MinMaxScaler()
    marked_for_drop = []

    @Variated(number_of_variations=200)
    def generate(self, V: Variat, input_data: DataFrame, is_training: bool):

        self.marked_for_drop = ['Title','Parch', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Age', 'Fare']

        transformed_data = input_data.copy()
        if is_training == True:
            # todo make class as a const or something less magiic stringy
            transformed_data.rename(columns={'Survived': 'class'}, inplace=True)

        transformed_data = self.transform_title(V, transformed_data, input_data)
        transformed_data = self.transform_sex(V, transformed_data, input_data)
        transformed_data = self.transform_embarked(V, transformed_data, input_data)
        transformed_data = self.transform_family_size(V, transformed_data, input_data)
        transformed_data = self.transform_is_alone(V, transformed_data, input_data)
        transformed_data = self.transform_name_length(V, transformed_data, input_data)
        transformed_data = self.transform_pclass(V, transformed_data, input_data)
        transformed_data = self.transform_fare(V, transformed_data, input_data)
        transformed_data = self.transform_age(V, transformed_data, input_data)
        transformed_data = self.transform_csv_postprocess(V, transformed_data, input_data)

        feature_layer = DataFrameHelper.columns_to_dense_features_input_layer(transformed_data.columns.tolist())
        borova_frame = DataFrameWithFeatures(feature_layer, transformed_data)

        yield borova_frame


    def transform_title(self, V: Variat, transformed: DataFrame, input: DataFrame):
        transformed['Title'] = input.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        transformed['Title'] = transformed['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                               'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        transformed['Title'] = transformed['Title'].replace('Mlle', 'Miss')
        transformed['Title'] = transformed['Title'].replace('Ms', 'Miss')
        transformed['Title'] = transformed['Title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        transformed['Title'] = transformed['Title'].map(title_mapping)
        transformed['Title'] = transformed['Title'].fillna(0)

        if V.B('title_scaling_to_one', True):
            transformed['Title_Scaled'] = self.scaler.fit_transform(transformed[['Title']])
        if V.B('title_one_hot_encode', True):
            one_hot_title_columns = pd.get_dummies(transformed['Title'], prefix='title')
            transformed = pd.concat([transformed, one_hot_title_columns], axis=1)

        return transformed

    def transform_sex(self, V: Variat, transformed: DataFrame, input: DataFrame):
        transformed['Sex'] = input['Sex'].map({'male': 0, 'female': 1})
        return transformed

    def transform_embarked(self, V: Variat, transformed: DataFrame, input: DataFrame):
        transformed['Embarked'] = transformed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

        freq_port = transformed.Embarked.dropna().mode()[0]
        transformed['Embarked'] = transformed['Embarked'].fillna(freq_port)
        transformed['Embarked'] = transformed['Embarked'].astype(int)

        transformed['Embarked_Scalled'] = self.scaler.fit_transform(transformed[['Embarked']])

        return transformed

    def transform_family_size(self, V: Variat, transformed: DataFrame, input: DataFrame):
        transformed['FamilySize'] = input['SibSp'] + input['Parch'] + 1
        return transformed

    def transform_is_alone(self, V: Variat, transformed: DataFrame, input: DataFrame):
        transformed['IsAlone'] = 0
        transformed.loc[transformed['FamilySize'] == 1, 'IsAlone'] = 1
        return transformed

    def transform_name_length(self, V: Variat, transformed: DataFrame, input: DataFrame):
        transformed['NameLength'] = input['Name'].str.len()
        transformed['NameLength_Scalled'] = self.scaler.fit_transform(transformed[['NameLength']])
        transformed = transformed.drop(['NameLength'], axis=1)
        return transformed

    def transform_pclass(self, V: Variat, transformed: DataFrame, input: DataFrame):
        transformed['Pclass'] = self.scaler.fit_transform(transformed[['Pclass']])
        transformed['Pclass'] = transformed['Pclass'].astype(int)
        return transformed

    def transform_fare(self, V: Variat, transformed: DataFrame, input: DataFrame):
        transformed['Fare_Scaled'] = self.scaler.fit_transform(transformed[['Fare']])
        transformed['Fare_Bucketized'] = pd\
            .cut(input['Fare'], [0, V.R("fr1", 6, 9), V.R("fr2", 12, 16), V.R("fr3", 27, 35), 9000]).cat.codes
        return transformed

    def transform_age(self, V: Variat, transformed: DataFrame, input: DataFrame):
        def generate_missing_age(data):
            for index, row in data.iterrows():
                if math.isnan(row['Age']):
                    curr_title = data.at[index, 'Title']
                    curr_pclass = data.at[index, 'Pclass']
                    maching_group = data.loc[
                        (data["Title"] == curr_title) & (data["Pclass"] == curr_pclass), "Age"]
                    mean_age = maching_group.mean()
                    std_age = maching_group.std()
                    normal_guess = np.random.normal(mean_age, std_age)
                    data.at[index, 'Age'] = normal_guess
            data['Age'] = data['Age'].astype(int)

        generate_missing_age(transformed)
        transformed['Age_Scalled'] = self.scaler.fit_transform(transformed[['Age']])
        transformed['Age_Bucketized'] = pd\
            .cut(transformed['Age'], [0, V.R("ag1", 7, 13), V.R("ag2", 15, 23), V.R("ag3", 25, 31), V.R("ag4", 35, 44), 50, 65, 200]).cat.codes

        return transformed

    def transform_csv_postprocess(self, V: Variat, transformed: DataFrame, input: DataFrame):
        return transformed.drop(self.marked_for_drop, axis=1)


mark1 = BorovaExperiment()
mark1.setup(
    input_data_processor=Mark1_InputDataProcessor(),
    models_provider=Mark1_ModelsProvider(),
    optimizers_provider=Mark1_OptimizersProvider(),
    losses_provider=Mark1_LossProvider()
)

evaluate = False

if evaluate:
    model_package_path = os.getcwd() + '\\network_models\\borova\\' + "20200728170746-mark001-0.8666666448116302"
    test_data_path = os.getcwd() + '\\input_data\\test.csv'
    y_values = pd.read_csv(test_data_path)['PassengerId']
    predicted_probabilities = mark1.validate_model(model_package_path, test_data_path, 'mark001-validation')
    predicted_classes = KaggleHelper.ConvertProbabilititsToClasses(pd.DataFrame(predicted_probabilities))
    KaggleHelper.CreateSubmission('borova', y_values.values, predicted_classes[0].values)
else:
    mark1.load_data('input_data/train.csv')
    mark1.run("mark001")