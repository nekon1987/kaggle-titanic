import pandas as pd

class KaggleHelper(object):

    @staticmethod
    def ConvertProbabilititsToClasses(survived_class: pd.DataFrame):
        for index, row in survived_class.iterrows():
            if survived_class.at[index, 0] > 0.5:
                survived_class.at[index, 0] = 1
            else:
                survived_class.at[index, 0] = 0

        survived_class[0] = survived_class[0].astype(int)
        return survived_class

    @staticmethod
    def CreateSubmission(file_name, passanger_ids, survived_class):
        submission = pd.DataFrame({
            "PassengerId": passanger_ids,
            "Survived": survived_class
        })

        submission.to_csv('kaggle_submission_files/' + file_name  + '.csv', index=False)
