from pandas import DataFrame
from tensorflow.python.feature_column.dense_features_v2 import DenseFeatures


class DataFrameWithFeatures(object):

    FeaturesLayer = DenseFeatures
    InputData = DataFrame

    def __init__(self, features_layer: DenseFeatures, input_data: DataFrame):
        self.FeaturesLayer = features_layer
        self.InputData = input_data