from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_linear_regression_model():
    lrm = make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )

    return lrm

