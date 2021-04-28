import pandas as pd
from sklearn.model_selection import train_test_split

def convert_long_lat_to_numeric(x):
    coord = float(x[:-1])
    emisphere = x[-1]

    if emisphere in ["N", "E"]:
        coord = -coord

    return coord

def prepare_dataset_global_temp_city(dataset: pd.DataFrame):
    '''
    function to prepare the Global Temperatures dataset for normalization
    :param dataset: pandas DataFrame containing the GlobalTemperatures all_dataset
    :return: None
    '''

    dataset = dataset.copy()

    # drop columns that are not needed
    dataset = dataset.drop(columns=['AverageTemperatureUncertainty', 'City',
                                    'Country'])

    # convert dt (date) column to DateTime object and add column for year
    dataset['dt'] = pd.to_datetime(dataset['dt'])
    dataset['Year'] = dataset['dt'].dt.year

    # drop old dt column
    dataset = dataset.drop('dt', axis=1)

    # drop all data before 1850 as there are missing values
    # dataset = dataset[dataset.Year >= 1850]

    # set index of dataset to Year column
    dataset = dataset.set_index(['Year'])

    # drop NaN values
    dataset = dataset.dropna(axis=0, how='any')

    dataset['Latitude'] = dataset['Latitude'].apply(convert_long_lat_to_numeric)
    dataset['Longitude'] = dataset['Longitude'].apply(convert_long_lat_to_numeric)

    return dataset


def separate_target_features_global_temp_city(dataset: pd.DataFrame):
    '''
    separate the features from the targets
    :param dataset: pandas DataFrame containing the dataset
    :return: feature matrix X and target vector Y
    '''

    # name of target column
    target = 'AverageTemperature'

    # target vector
    Y = dataset[target]

    # feature matrix
    X = dataset[['Latitude', 'Longitude']]

    return X, Y


def split_dataset_global_temp_city(X: pd.DataFrame, Y: pd.Series):
    '''
    split features and targets in train and test data and write them to files
    :param X: pandas DataFrame features matrix
    :param Y: pandas Series targets vector
    :return: None
    '''

    # split data - 80% train 20% test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # write data to files
    X_train.to_csv('./dataset/X_train2.csv')
    X_test.to_csv('./dataset/X_test2.csv')
    Y_train.to_csv('./dataset/Y_train2.csv')
    Y_test.to_csv('./dataset/Y_test2.csv')

