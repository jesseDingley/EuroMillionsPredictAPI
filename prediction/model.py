"""A module for training and testing euromillions prediction model.

This module contains X functions for preparing, training and testing.

    Typical usage example:

    X_train, X_test, y_train, y_test = split_data(df)
    model = train(X_train, y_train)
    f1 = performance_test(model, X_test, y_test)
    p = predict_combination(model, {"N1": 0, "N2": 0, "N3": 0, "N4": 0, "N5": 0, "E1": 0, "E2": 0})
"""


"""Imports."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


"""Hyperparameters."""
TEST_SIZE = 0.1
NUM_TREES = 1000


"""Meta data."""
PERFORMANCE_METRIC = "F1-score"
ALGORITHM = "Random Forest"


def split_data(df, test_size = TEST_SIZE):
    """Splits data into train and test splits.

    Args:
        df (pandas.core.frame.DataFrame): dataframe.
        test_size (float): test / train ratio.

    Returns:
        tuple of pandas.core.frame.DataFrame: (train data, test data, train labels, test labels).
    """

    #X: features, y: labels
    X = df[["N1","N2","N3","N4","N5","E1","E2"]]
    y = df["Winner_binary"]

    # split data into train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    return (X_train, X_test, y_train, y_test)


def train(X_train,y_train, num_trees = NUM_TREES):
    """Trains random forest model.

    Args:
        X_train (pandas.core.frame.DataFrame): training data.
        y_train (pandas.core.frame.DataFrame): training labels.

    Returns:
        sklearn.ensemble._forest.RandomForestClassifier: trained rf model.
    """

    # define model
    model_ = RandomForestClassifier(n_estimators = num_trees)

    # actually train the model
    model_.fit(X_train, y_train)

    return model_


def performance_test(trained_model, X_test, y_test):
    """Returns F1 score of model on test data.

    Args:
        trained_model (sklearn.ensemble._forest.RandomForestClassifier): trained rf model.
        X_test (pandas.core.frame.DataFrame): testing data.
        y_test (pandas.core.frame.DataFrame): testing labels.

    Returns:
        float: F1 score.
    """
    y_pred = trained_model.predict(X_test)
    return f1_score(y_test, y_pred, average = "binary")


def predict_combination(trained_model, combination):
    """Returns the probability of a combination being a win.
    
    Args:
        combination (dict): dictionary where keys are "N1", ..., "E2" and values are associated numbers.
        trained_model (sklearn.ensemble._forest.RandomForestClassifier): trained model

    Returns:
        float: probability of the combination being a win (0 < p < 1).
    """
    df_combination = pd.DataFrame.from_records(data = [combination])
    return trained_model.predict_proba(df_combination)[0][0]