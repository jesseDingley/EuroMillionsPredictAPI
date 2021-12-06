"""A module for preprocessing raw csv data.

This module contains four functions for preparing raw csv data for usage.

    Typical usage example:

    raw_df = create_df("../data/EuroMillions_numbers.csv")
    combination = generate_random_combination()
    new_df = add_data(raw_df, 10)
    add_binary_winner_column(new_df)
"""


import pandas as pd
import numpy as np


def create_df(file_path):
    """Returns a pandas dataframe from a csv.

    Args:
        file_path (string): File path of csv.

    Returns:
        pandas.core.frame.DataFrame.
    """
    return pd.read_csv(file_path, sep = ";")


def generate_random_combination():
    """Return a random combination.

    Generates a random combination by concatenating two sub arrays.
    This is because a combination has to follow the [N1, ... , N5, E1, E2] format
    where 1 <= N_i <= 50 and 1 <= E_i <= 12.

    Returns:
        numpy.ndarray: numpy array of random combination.
    """
    N = np.random.randint(low = 1, high = 50+1, size=5)
    E = np.random.randint(low = 1, high = 12+1, size=2)
    return np.concatenate((N,E))


def add_data(df, nb_new_combinations):
    """Appends a fixed number of new combinations to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): dataframe.
        nb_new_combinations (int): number of new combinations to add.
    
    Returns:
        pandas.core.frame.DataFrame: dataframe with new combinations added.
    """
    for i in range(nb_new_combinations):

        # calculate new combination i
        seen_combinations = df[["N1","N2","N3","N4","N5","E1","E2"]].to_numpy()
        candidate = generate_random_combination()
        while True:
            if candidate.tolist() in seen_combinations.tolist(): # need new candidate
                candidate = generate_random_combination()
            else: # candidate is a new one (does not already exist in dataframe)
                break

        # append new line i
        d = {"Date": "X", "N1": candidate[0], "N2": candidate[1], "N3": candidate[2], "N4": candidate[3], "N5": candidate[4], "E1": candidate[5], "E2": candidate[6], "Winner": 0, "Gain": 0}
        new_line = pd.Series(data = d)
        df = df.append(new_line, ignore_index = True)
    return df

def add_binary_winner_column(df):
    """Adds 'Winner_binary' column to dataframe.

    Adds new column to dataframe that corresponds to a binarization of the 'Winner' column.
    The new column indicates if the combination has one or multiple winners.
    The 'Winner' column has for example values 0, 1, 5, 2 (number of winners with the given combination).
    We replace for example 5 by 1, so we can perform binary classification (combination is winner or not).

    Args:
        df (pandas.core.frame.DataFrame): dataframe
    """
    df["Winner_binary"] = df["Winner"].apply(lambda x: 1 if x >= 1 else 0)