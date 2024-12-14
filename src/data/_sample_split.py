import pandas as pd
import hashlib
import numpy as np

def create_sample_split(df, id_column, training_frac=0.8):
    """Create a train-test split for a dataset based on a hash or modulo of an ID column.

    This function assigns each record in the dataset to either the training or testing set
    based on the values in a specified ID column. It ensures the split is consistent and reproducible.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be split.
    id_column : str
        The name of the column containing unique identifiers for splitting.
    training_frac : float, optional
        The fraction of data to include in the training set (default is 0.8).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column named "sample", indicating whether
        each record belongs to the "train" or "test" set.

    Notes
    -----
    - For integer ID columns, the modulo operation is used to determine the split.
    - For non-integer ID columns, an MD5 hash function is applied to ensure consistency.
    - The split is based on the training fraction multiplied by 100 (e.g., 80% = training_frac * 100).
    """

    if df[id_column].dtype == np.int64:
        modulo = df[id_column] % 100
    else:
        modulo = df[id_column].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 100
        )

    df["sample"] = np.where(modulo < training_frac * 100, "train", "test")

    return df
