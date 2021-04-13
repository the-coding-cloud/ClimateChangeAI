import pandas as pd


def read_dataset(dataset_path: str) -> pd.DataFrame:
    '''
    read the all_dataset from csv file into pandas DataFrame
    :param dataset_path: path to all_dataset
    :return: pandas DataFrame containing the all_dataset
    '''
    return pd.read_csv(dataset_path)
