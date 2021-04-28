from preprocessing.preprocess_dataset_global_temperatures_city import prepare_dataset_global_temp_city, separate_target_features_global_temp_city, split_dataset_global_temp_city
from utils.utils import read_dataset


if __name__ == '__main__':
    dataset = read_dataset('./dataset/GlobalLandTemperaturesByMajorCity.csv')
    print(dataset.shape)
    print(dataset.columns)

    print('\n')

    dataset = prepare_dataset_global_temp_city(dataset)
    print(dataset.shape)
    print(dataset.columns)
    print(dataset.head())

    X, Y = separate_target_features_global_temp_city(dataset)

    split_dataset_global_temp_city(X, Y)