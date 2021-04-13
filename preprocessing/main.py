from preprocessing.preprocess_dataset_global_temperatures import prepare_dataset_global_temperatures, \
    separate_target_features_global_temperatures, split_dataset_global_temperatures
from utils.utils import read_dataset

if __name__ == '__main__':
    dataset = read_dataset('./dataset/GlobalTemperatures.csv')
    print(dataset.shape)
    print(dataset.columns)

    print('\n')

    dataset = prepare_dataset_global_temperatures(dataset)
    print(dataset.shape)
    print(dataset.columns)
    print(dataset.head())

    X, Y = separate_target_features_global_temperatures(dataset)

    split_dataset_global_temperatures(X, Y)