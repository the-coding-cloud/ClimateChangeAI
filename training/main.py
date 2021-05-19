from utils.utils import read_dataset
from training.linear_regression import make_linear_regression_model
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__':
    X_train = read_dataset("../preprocessing/dataset/X_train.csv")
    X_test = read_dataset("../preprocessing/dataset/X_test.csv")
    Y_train = read_dataset("../preprocessing/dataset/Y_train.csv")
    Y_test = read_dataset("../preprocessing/dataset/Y_test.csv")

    X_train = X_train.drop(X_train.columns[0], axis=1)
    X_test = X_test.drop(X_test.columns[0], axis=1)

    Y_train = Y_train.drop(Y_train.columns[0], axis=1)
    Y_test = Y_test.drop(Y_test.columns[0], axis=1)

    pred_baseline = [Y_train.mean()] * len(Y_train)

    print('Baseline mean absolute error:', round(mean_absolute_error(Y_train, pred_baseline), 5))

    lrm = make_linear_regression_model()

    lrm.fit(X_train, Y_train)

    print('Linear regression - Training MAE:', round(mean_absolute_error(Y_train, lrm.predict(X_train)), 5))
    print('Linear regression - Test MAE', round(mean_absolute_error(Y_test, lrm.predict(X_test)), 5))


