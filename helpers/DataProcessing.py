from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.DataframeHelper import DataframeHelper

class DataProcessing(object):

    @staticmethod
    def run_complete_pipeline_transformations(data_csv, is_training):
        DataProcessing.describe_input_csv(data_csv)
        DataProcessing.transform_csv_preprocessing(data_csv, is_training)
        DataProcessing.transform_csv_create_title(data_csv)
        DataProcessing.transform_csv_generate_missing_age(data_csv)
        DataProcessing.transform_csv_process_embarked(data_csv)
        DataProcessing.transform_csv_categorical_values(data_csv)
        DataProcessing.transform_csv_scale_to_one(data_csv)
        csv_processed = DataProcessing.transform_csv_postprocess(data_csv)
        DataProcessing.perform_visual_analysis(csv_processed)
        return csv_processed

    @staticmethod
    def describe_input_csv(data):
        print(data.describe(include='all'))
        print(data.describe(include=['O']))
        print(data.isnull().sum())
        print('-----------------------------------------------------')

    @staticmethod
    def transform_csv_scale_to_one(data):
        scaler = MinMaxScaler()
        data['NameLength'] = scaler.fit_transform(data[['NameLength']])
        data['Pclass'] = scaler.fit_transform(data[['Pclass']])
        data['Fare'] = scaler.fit_transform(data[['Fare']])
        data['Age'] = scaler.fit_transform(data[['Age']])
        data['Title'] = scaler.fit_transform(data[['Title']])
        data['Embarked'] = scaler.fit_transform(data[['Embarked']])

    @staticmethod
    def transform_csv_preprocessing(data, isTraining):
        if isTraining == True:
            data.rename(columns={'Survived': 'class'}, inplace=True)  # tpot needs this as Y

        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
        data['NameLength'] = data['Name'].str.len()
        print('-----------------------------------------------------')

    @staticmethod
    def transform_csv_create_title(data):
        data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        data['Title'] = data['Title'].map(title_mapping)
        data['Title'] = data['Title'].fillna(0)

    @staticmethod
    def transform_csv_categorical_values(data):
        data['Fare'] = pd.cut(data['Fare'], [0,7.911,14.455,31.01,90000]).cat.codes
        data['Age'] = pd.cut(data['Age'], [0, 10, 18, 26, 36, 50, 65, 200]).cat.codes

    @staticmethod
    def transform_csv_generate_missing_age(data):
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

    @staticmethod
    def transform_csv_process_embarked(data):
        freq_port = data.Embarked.dropna().mode()[0]
        data['Embarked'] = data['Embarked'].fillna(freq_port)
        data['Embarked'] = data['Embarked'].astype(int)

    @staticmethod
    def transform_csv_postprocess(data):
        return data.drop(['FamilySize','SibSp','Parch','Name','Ticket','Cabin','PassengerId'], axis=1)

    @staticmethod
    def get_test_and_validation(csv_processed):
        train, test = train_test_split(csv_processed, train_size=0.80, test_size=0.20)
        batch_size = 892
        train_ds = DataframeHelper.df_to_dataset(train, batch_size=batch_size)
        val_ds = DataframeHelper.df_to_dataset(test, batch_size=batch_size)
        return train_ds, val_ds

    @staticmethod
    def perform_visual_analysis(data):
        plt.figure(figsize=(30, 12))
        sns.heatmap(data.corr(), vmax=0.6, square=True, annot=True)
        plt.show()
        print('-----------------------------------------------------')
