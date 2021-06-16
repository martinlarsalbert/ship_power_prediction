from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import pandas as pd

def get_train_test(data:pd.DataFrame, rolling=True, label='Power', test_size=0.2, random_state=42):
    """[summary]

    Args:
        data ([type]): [description]
        rolling (bool, optional): rolling basis or random?. Defaults to True.
        label (str, optional): [description]. Defaults to 'Power'.

    Returns:
        [type]: X_train, X_test, y_train, y_test 
    """

    if rolling:
        return get_train_test_rolling(data=data, label=label, test_size=test_size)
    else:
        X,y = get_X_y(data=data, label=label)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_cross_validation(n_splits=6, n_repeats=10, rolling=True, shuffle=False):
    if rolling:
        return get_cross_validation_rolling(n_splits=n_splits, n_repeats=n_repeats)
    else:
        if shuffle:
            return KFold(n_splits=n_splits, random_state=42, shuffle=shuffle)
        else:
            return KFold(n_splits=n_splits, shuffle=shuffle)

def get_X_y(data:pd.DataFrame, label:str):
    X = data.copy()
    y = X.pop(label)
    return X,y


def get_train_test_rolling(data, label='Power', test_size=0.2, **kwargs):
    X,y = get_X_y(data=data, label=label)

    split_index = int(len(X)*(1-test_size))
    X_train = X.loc[0:split_index].copy()
    y_train = y.loc[0:split_index].copy()

    X_test = X.loc[split_index:].copy()
    y_test = y.loc[split_index:].copy()

    return X_train, X_test, y_train, y_test

def get_cross_validation_rolling(n_splits=6, **kwargs):
    cv = TimeSeriesSplit(n_splits=n_splits)
    return cv
