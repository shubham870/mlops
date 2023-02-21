import re

import numpy as np
import pandas as pd


def _get_first_cabin(row):
    try:
        return row.split()[0]
    except Exception:
        return np.nan


def _get_title(passenger) -> str:
    line = passenger
    if re.search("Mrs", line):
        return "Mrs"
    elif re.search("Mr", line):
        return "Mr"
    elif re.search("Miss", line):
        return "Miss"
    elif re.search("Master", line):
        return "Master"
    else:
        return "Other"


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    new_df = dataframe.copy()

    new_df = new_df.replace("?", np.nan)
    new_df["cabin"] = new_df["cabin"].apply(_get_first_cabin)
    new_df["title"] = new_df["name"].apply(_get_title)
    new_df["fare"] = new_df["fare"].astype("float")
    new_df["age"] = new_df["age"].astype("float")
    new_df.drop(
        labels=["name", "ticket", "boat", "body", "home.dest"], axis=1, inplace=True
    )

    return new_df
