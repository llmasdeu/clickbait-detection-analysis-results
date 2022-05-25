import json
import pprint
import numpy as np
import pandas as pd
import seaborn as sn

files = ['headlines_001.json', 'headlines_002.json', 'headlines_003.json', 'headlines_004.json', 'headlines_005.json',
         'headlines_006.json']

def get_json_content(i):
    # Checks if the index is valid
    if i < 0 or i >= len(files):
        return {}

    # Gets the data from the JSON file
    data = read_json_file(f'./files/{files[i]}')

    # Sets lists for the headline and the target
    X = data['headline'].to_numpy()
    Y = data['is_clickbait'].to_numpy()

    # Returns the headline and the target
    return X, Y

def read_json_file(file):
    # Reads the JSON file
    df = pd.read_json(file)

    # Changes the booleans to integer
    df['is_clickbait'] = df['is_clickbait'].astype(int)
    df['original_dataset'] = df['original_dataset'].astype(int)

    # Returns the dataframe
    return df

def permutate_learner_parameters(params):
    # Gets the keys
    keys = list(params.keys())

    # Gets the values
    values = list(params.values())

    # Sets a temporal list
    matrix = []

    # Iterates through the keys
    for i in range(len(keys)):
        # Sets the current list of combinations
        cur_list = []

        # Iterates through the values
        for j in range(len(values[i])):
            # Appends the current combination
            cur_list.append({keys[i]: values[i][j]})

        # Appends the current combinations
        matrix.append(cur_list)

    # Sets the definitive list
    y = []

    # Iterates through the list of combinations
    for i in matrix[0]:
        for j in matrix[1]:
            # Appends the current set of combinations
            y.append(dict(i.items() | j.items()))

    # Returns the definitive list of combinations
    return y