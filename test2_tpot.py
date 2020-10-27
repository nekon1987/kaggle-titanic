from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from helpers.DataProcessing import DataProcessing
from helpers.KaggleHelper import KaggleHelper

test_csv = pd.read_csv('input_data/test.csv')
train_data_csv = DataProcessing.run_complete_pipeline_transformations(pd.read_csv('input_data/train.csv'), True)
test_data_csv = DataProcessing.run_complete_pipeline_transformations(test_csv, False)
passenger_ids = test_csv['PassengerId']


tpot = TPOTClassifier(config_dict='TPOT NN', verbosity=2, max_time_mins=0.5, max_eval_time_mins=4, population_size=150)

labels = train_data_csv.pop('class')
train_x, validation_x, train_y, validation_y = train_test_split(train_data_csv, labels, train_size=0.85, test_size=0.15)

tpot.fit(train_x, train_y)
score = tpot.score(validation_x, validation_y)

print(score)
tpot.export('network_models/tpot/' + time.strftime("%Y%m%d-%H%M%S") + ' score-' + str(score) + '.py')

predicted_classes = tpot.predict(test_data_csv.to_numpy())
KaggleHelper.CreateSubmission('tpot', passenger_ids.values, predicted_classes)

print(time.strftime("%Y%m%d-%H%M%S") + ' all done!')









#
#
# # Id - Y splits
# training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.80, test_size=0.20)
#
# tpot = TPOTClassifier(verbosity=2, max_time_mins=5, max_eval_time_mins=0.7, population_size=130)
# features = titanic_new[training_indices]
# target=titanic_class[training_indices]
#
# tpot.fit(features, target)
#
# print(tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values))
# tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values)
