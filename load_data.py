"""
This script responsible for everything about loading the data.

All data's parameters are also need to be in this script.
"""

import pandas as pd

data_params = {
    'paths': {'measurements': 'data/measurements.txt',
              'controls': 'data/controls.txt',
              'ground_truth': 'data/ground_truth.txt'
              },
    'columns': {'measurements': ['time', 'r', 'phi'],
                'controls': ['time', 'v', 'omega'],
                'ground_truth': ['time', 'x', 'y', 'theta']
                }
}


def load_data():
    """
    This function get the data and return it as dict

    :return: dict with all the require data
    """
    paths = data_params['paths']
    columns = data_params['columns']
    data = {}

    for file in paths:
        data[file] = pd.read_csv(paths[file], names=columns[file])

    return data