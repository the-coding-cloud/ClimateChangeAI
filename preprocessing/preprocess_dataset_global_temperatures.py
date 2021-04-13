import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_dataset_global_temperatures(dataset: pd.DataFrame):
    '''
    function to prepare the Global Temperatures dataset for normalization
    :param dataset: pandas DataFrame containing the GlobalTemperatures all_dataset
    :return: None
    '''

    dataset = dataset.copy()

    # drop columns that are not needed
    dataset = dataset.drop(columns=['LandAverageTemperatureUncertainty', 'LandMaxTemperatureUncertainty',
                                    'LandMinTemperatureUncertainty', 'LandAndOceanAverageTemperatureUncertainty'])

    # convert dt (date) column to DateTime object and add column for year
    dataset['dt'] = pd.to_datetime(dataset['dt'])
    dataset['Year'] = dataset['dt'].dt.year

    # drop old dt column
    dataset = dataset.drop('dt', axis=1)

    # drop all data before 1850 as there are missing values
    dataset = dataset[dataset.Year >= 1850]

    # set index of dataset to Year column
    dataset = dataset.set_index(['Year'])

    # drop NaN values
    dataset = dataset.dropna()

    return dataset


def separate_target_features_global_temperatures(dataset: pd.DataFrame):
    '''
    separate the features from the targets
    :param dataset: pandas DataFrame containing the dataset
    :return: feature matrix X and target vector Y
    '''

    # name of target column
    target = 'LandAndOceanAverageTemperature'

    # target vector
    Y = dataset[target]

    # feature matrix
    X = dataset[['LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']]

    return X, Y


def split_dataset_global_temperatures(X: pd.DataFrame, Y: pd.Series):
    '''
    split features and targets in train and test data and write them to files
    :param X: pandas DataFrame features matrix
    :param Y: pandas Series targets vector
    :return: None
    '''

    # split data - 80% train 20% test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # write data to files
    X_train.to_csv('./dataset/X_train.csv')
    X_test.to_csv('./dataset/X_test.csv')
    Y_train.to_csv('./dataset/Y_train.csv')
    Y_test.to_csv('./dataset/Y_test.csv')



