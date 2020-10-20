import pandas as pd
import numpy as np

def one_hot_encode(df, column_name, prefix = '', replace_column = True, insert_to_end = False):
    """
    Performs one hot encoding on the given column in the data and replaces this column with the
    new one hot encoded columns
    :param df: The data frame in question
    :param column_name: The column to one hot encode
    :param prefix: (Optional, Default: column_name) The prefix for the new columns
    :param replace_column: (Optional, Default: True) Whether or not to replace the column to encode
    :param insert_to_end: (Optional, Default: False) Whether or not to add encoded columns at the end
    :return: The same data frame with the specified changes
    """

    dummies_insertion_index = df.columns.get_loc(column_name)
    dummies = pd.get_dummies(df[column_name], prefix=column_name if prefix == '' else prefix)

    if replace_column:
        df = df.drop([column_name], axis=1)
    else:
        dummies_insertion_index += 1

    if insert_to_end:
        df = pd.concat([df, dummies], axis=1)
    else:
        for column_to_insert in dummies.columns:
            df.insert(loc=dummies_insertion_index, column=column_to_insert, value=dummies[column_to_insert])
            dummies_insertion_index += 1

    return df

def read_csv(file_path, verbose=True):
    """
    Reads a csv file and returns the smallest possible dataframe
    :param file_path: The file path
    :param verbose: Whether or not to be verbose about the memory savings
    :return: An optimized dataframe
    """

    ret_val = pd.read_csv(file_path)
    return reduce_mem_usage(ret_val, verbose)

def reduce_mem_usage(df, verbose=True):
    """
    Takes a dataframe and returns one that takes the least memory possible.
    This works by going over each column and representing it with the smallest possible data structure.
    Example usage: my_data = pd.read_csv('D:/SomeFile.csv').pipe(reduce_mem_usage)
    Source: (https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65)
    Parameters:
        df (DataFrame): The dataframe to optimize
        verbose (bool): Whether or not to be verbose about the savings
    """

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df