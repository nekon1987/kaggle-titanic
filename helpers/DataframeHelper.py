import tensorflow as tf

# https://github.com/Vincent0700/tcplite
from pandas import DataFrame
from sklearn.model_selection import train_test_split


class DataframeHelper(object):

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    @staticmethod
    def df_to_dataset(dataframe, shuffle=False, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('class')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    @staticmethod
    def get_test_and_validation(csv_processed: DataFrame, train_size: float, test_size: float):
        train, test = train_test_split(csv_processed, train_size=train_size, test_size=test_size)
        batch_size = 892
        train_ds = DataframeHelper.df_to_dataset(train, batch_size=batch_size)
        val_ds = DataframeHelper.df_to_dataset(test, batch_size=batch_size)
        return train_ds, val_ds
