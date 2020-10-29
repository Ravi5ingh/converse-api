import pandas as pd
import numpy as np
import os as os
import requests as rq
import pickle as pkl
import gensim as gs
import sys as sy
import sqlalchemy as sq

def read_db(database_filename, table_name):
    """
    Read a db file and return it as a dataframe
    :param database_filename: The DB file path
    :param table_name: The table name
    :return: The dataframe
    """

    engine = sq.create_engine('sqlite:///' + database_filename)
    return pd.read_sql(table_name, con=engine)

def to_db(df, database_filename, table_name, index = False):
    """
    Save a data frame as a SQLite DB file to the given location with the given table name
    :param df: The data frame to save
    :param database_filename: The DB file to create (NOTE: Will be replaced if it exists)
    :param index: (Optional, Default: False) Whether or not to create an index column in the saved table
    :param table_name: The name of the table to contain the data frame data
    """

    # If the DB file exists, delete it
    if os.path.exists(database_filename):
        os.remove(database_filename)

    # Save data to an sqlite db
    engine = sq.create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=index)

def widen_df_display():
    """
    Widens the way dataframes are printed (setting lifetime is runtime)
    """

    pd.set_option('display.width', 3000)
    pd.set_option('display.max_columns', 100)

def try_word2vec(word):
    """
    Gets the word vector for the given work based on Google's trained model.
    1. Tries the cache first
    2. Loads the model between 0 and 1 times per run (Will download it automatically if necessary)
    3. Updates cache
    :param word: The word to vectorize
    :return: The (word vector, boolean for success/failure)
    """

    global google_word2vec_model
    global word2vec_cache
    model_filename = __file__[0:__file__.rindex('\\')] + '\\..\\models\\nl\\GoogleWord2VecModel.bin'
    cache_filename = __file__[0:__file__.rindex('\\')] + '\\..\\models\\nl\\word2vec_cache.pkl'

    # Check cache
    if word2vec_cache is None:
        if os.path.exists(cache_filename):
            word2vec_cache = read_pkl(cache_filename)
        else:
            word2vec_cache = {}
            to_pkl(word2vec_cache, cache_filename)

    # Try cache
    if word in word2vec_cache:
        return word2vec_cache[word], word2vec_cache[word] is not None
    # Use Google's model
    else:
        if google_word2vec_model is None:
            print('Need to load Google word2vec Model')

            # Check if model exists, download otherwise
            if not os.path.exists(model_filename):
                print('Google word2vec model not found. Will download (~3.5GB)...')
                download_gdrive_file('1kzCpXqZ_EILFAfK4G96QZBrjtezxjMiO', model_filename) # Hard-coded file id
                print('Done downloading Google word2vec model')

            print('Loading Google word2vec model...')
            google_word2vec_model = gs.models.KeyedVectors.load_word2vec_format(model_filename, binary=True)
            print('Done loading Google word2vec model')

        try:
            word2vec_cache[word] = google_word2vec_model[word]
            to_pkl(word2vec_cache, cache_filename)
            return word2vec_cache[word], True
        except:
            word2vec_cache[word] = None
            to_pkl(word2vec_cache, cache_filename)
            return word2vec_cache[word], False

def read_pkl(file_name):
    """
    De-serializes a pickle file into an object and returns it
    :param file_name: The name of the pickle file
    :return: The object that is de-serialized
    """

    with open(file_name, 'rb') as file:
        return pkl.load(file)

def to_pkl(obj, file_name):
    """
    Save the given object as a pickle file to the given file name
    :param obj: The object to serialize
    :param file_name: The file name to save it to
    :return: returns the same object back
    """

    with open(file_name, 'wb') as file:
        pkl.dump(obj, file)

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

def download_gdrive_file(file_id, output_file_path):
    """
    Download a file from Google Drive given its file id
    (Source: https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python)
    :param file_id: The file id
    :param output_file_path: The path of the output file
    """

    URL = "https://docs.google.com/uc?export=download"

    session = rq.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = __get_confirm_token__(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    __save_response_content__(response, output_file_path)

def whats(thing) :
    """
    Prints the type of object passed in
    Parameters:
        thing (Object): The object for which the type needs to be printed
    """

    print(type(thing))

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

#region Properties

google_word2vec_model = None

word2vec_cache = None

#endregion

#region Private

def __get_confirm_token__(response):
    """
    Get a confirmation token from Google Drive (that says I'm ok with not scanning for viruses)
    :param response: The HTTP response object
    :return: The token
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def __save_response_content__(response, output_file_name):
    """
    Given an HTTP response object and a output file name, save the content to the file
    :param response: The HTTP response object
    :param output_file_name: The path of the output file
    """

    CHUNK_SIZE = 32768
    file_size = int(response.headers.get('Content-Length')) if response.headers.get('Content-Length') else None

    with open(output_file_name, "wb") as f:
        i = 1
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                mb_sofar = CHUNK_SIZE * i / 1024 / 1024
                if file_size:
                    percentage = (CHUNK_SIZE * i / file_size * 100)
                    sy.stdout.write('\r' + '[                                                  ]'
                                     .replace(' ', ':', int(percentage / 2)) + ' ' + str(
                        min(int(percentage), 100)) + '% (' + str(round(mb_sofar, 2)) + 'MB)')
                else:
                    sy.stdout.write('\r' + 'Unknown file size. ' + str(round(mb_sofar, 2)) + 'MB downloaded')
                f.write(chunk)
                i += 1
    print('')

#endregion