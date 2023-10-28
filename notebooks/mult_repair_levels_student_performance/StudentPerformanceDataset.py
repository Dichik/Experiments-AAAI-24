import pandas as pd
from virny.datasets.base import BaseDataLoader
from sklearn import preprocessing

class StudentPerformanceDataset(BaseDataLoader):

    def __init__(self, dataset_path=None):
        # if dataset_path is None:
        #     filename = 'ricci_race.csv'
        #     dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

        df = pd.read_csv(dataset_path, delimiter=';')

        target = 'G3'
        df[target] = (df[target] >= 10) * 1

        # df['G1'] = (df['G1'] >= 10) * 1
        # df['G2'] = (df['G2'] >= 10) * 1

        df.drop(columns=['G1', 'G2'], inplace=True)

        categorical_columns = ['Mjob', 'Fjob', 'reason', 'guardian', 'sex']
        numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                             'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                             'absences']

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )