import tensorflow as tf

# https://github.com/Vincent0700/tcplite
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.python import feature_column


class DataFrameHelper(object):

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    @staticmethod
    def df_to_dataset(data_frame, shuffle=False, batch_size=32):
        data_frame = data_frame.copy()
        labels = data_frame.pop('class')
        ds = tf.data.Dataset.from_tensor_slices((dict(data_frame), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(data_frame))
        ds = ds.batch(batch_size)
        return ds

    @staticmethod
    def get_test_and_validation(csv_processed: DataFrame, train_size: float, test_size: float):
        train, test = train_test_split(csv_processed, train_size=train_size, test_size=test_size)
        batch_size = 892
        train_ds = DataFrameHelper.df_to_dataset(train, batch_size=batch_size)
        val_ds = DataFrameHelper.df_to_dataset(test, batch_size=batch_size)
        return train_ds, val_ds

    @staticmethod
    def columns_to_dense_features_input_layer(input_data_columns):
        if 'class' in  input_data_columns:
            input_data_columns.remove('class')

        feature_columns = []
        for header in input_data_columns:
            feature_columns.append(feature_column.numeric_column(header))

        return tf.keras.layers.DenseFeatures(feature_columns)