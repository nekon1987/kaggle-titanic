import pandas as pd
import time
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
from helpers.DataframeHelper import DataframeHelper
from helpers.DataProcessing import DataProcessing
from helpers.KaggleHelper import KaggleHelper
import matplotlib.pyplot as plt

improvement_found = 0
best_loss_score = 0
loss_scores = []
best_model = None
test_csv = pd.read_csv('input_data/test.csv')
train_data_csv = DataProcessing.run_complete_pipeline_transformations(pd.read_csv('input_data/train.csv'), True)
test_data_csv = DataProcessing.run_complete_pipeline_transformations(test_csv, False)
passenger_ids = test_csv['PassengerId']
train_ds, validation_ds = None, None

for a in range(1000):

    def create_input_features_layer():
        # todo: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/feature_columns.ipynb#scrollTo=r1tArzewPb-b
        feature_columns = []
        for header in ['Sex', 'IsAlone', 'NameLength', 'Embarked', 'Title', 'Pclass', 'Fare', 'Age']:
            feature_columns.append(feature_column.numeric_column(header))
        return feature_columns

    def get_test_and_validation(csv_processed, train_size, test_size):
        train, test = train_test_split(csv_processed, train_size=train_size, test_size=test_size)
        batch_size = 892
        train_ds = DataframeHelper.df_to_dataset(train, batch_size=batch_size)
        val_ds = DataframeHelper.df_to_dataset(test, batch_size=batch_size)
        return train_ds, val_ds


    feature_layer = tf.keras.layers.DenseFeatures(create_input_features_layer())
    model = tf.keras.Sequential([
      feature_layer,
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

    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    train_data_csv = DataProcessing.run_complete_pipeline_transformations(pd.read_csv('input_data/train.csv'), True)
    test_data_csv = DataProcessing.run_complete_pipeline_transformations(pd.read_csv('input_data/test.csv'), False)

    train_ds, validation_ds = get_test_and_validation(train_data_csv, train_size=0.75, test_size=0.25)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=420, verbose=0)
    loss_score = 2 - (history.history['loss'][-1] + history.history['val_loss'][-1])
    #loss_score = (history.history['accuracy'][-1] + history.history['val_accuracy'][-1])/ 2

    loss_scores.append(loss_score)
    plt.plot(loss_scores)
    plt.show()

    if loss_score > best_loss_score:
        best_loss_score = loss_score
        best_model = model
        improvement_found = improvement_found + 1
        print('***************** Found better model: ' + str(best_loss_score))
    else:
        print('***************** No improvement: ' + str(a))

best_model.save('network_models/dnn/' + time.strftime("%Y%m%d-%H%M%S") + ' lossscore-' + str(best_loss_score))

test_ds = tf.data.Dataset.from_tensor_slices(dict(test_data_csv))
test_ds = test_ds.batch(892)
predicted_probabilities = best_model.predict(test_ds)

eval_validation = best_model.evaluate(validation_ds)
eval_train = best_model.evaluate(train_ds)

predicted_classes = KaggleHelper.ConvertProbabilititsToClasses(pd.DataFrame(predicted_probabilities))
KaggleHelper.CreateSubmission('dnn', passenger_ids.values, predicted_classes[0].values)

print(time.strftime("%Y%m%d-%H%M%S") + ' all done!' + 'improved: ' + str(improvement_found) + ' with final score: ' + str(best_loss_score))