from sklearn.model_selection import train_test_split
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from helpers.DataProcessing import DataProcessing
from helpers.KaggleHelper import KaggleHelper
import time

test_csv = pd.read_csv('input_data/test.csv')
train_data_csv = DataProcessing.run_complete_pipeline_transformations(pd.read_csv('input_data/train.csv'), True)
test_data_csv = DataProcessing.run_complete_pipeline_transformations(test_csv, False)
passenger_ids = test_csv['PassengerId']

h2o.init()
model = H2OAutoML(max_models= 10, seed= 7, nfolds= 10)

train, validation,= train_test_split(train_data_csv, train_size=0.85, test_size=0.15)
feature_column_names = list(train.columns)
feature_column_names.remove('class')

train_h2o = h2o.H2OFrame(train_data_csv)
validation_h2o = h2o.H2OFrame(validation)
model.train(x=feature_column_names, y='class', training_frame=train_h2o, validation_frame=validation_h2o)

model.leaderboard
model.leader.model_performance(validation_h2o)

prediction_frame  = model.leader.predict(h2o.H2OFrame(test_data_csv))
predictions = prediction_frame[0].as_data_frame().values.flatten()
predicted = pd.DataFrame(predictions)

predicted_classes = KaggleHelper.ConvertProbabilititsToClasses(predicted)
KaggleHelper.CreateSubmission('h2o', passenger_ids.values, predicted_classes[0].values)

# save the model
model_path = h2o.save_model(model=model.leader, path="network_models/h2o", force=True)

print(time.strftime("%Y%m%d-%H%M%S") + ' all done!')





