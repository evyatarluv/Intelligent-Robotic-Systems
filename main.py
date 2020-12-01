import pandas as pd
import numpy as np
from figures import ground_truth_xy, ground_truth_subplots, ground_truth_measurements
from kalman_filter import kalman_filter

# Parameters


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


def load_data(paths, columns):
    """
    This function get the data and return it as dict

    :param columns: dict with the columns names for each data
    :param paths: dict with the data name as key and path as value
    :return:
    """
    data = {}

    for file in paths:
        data[file] = pd.read_csv(paths[file], names=columns[file])

    return data


def plot_figures(ground_truth, measurements):

    # Figure 1
    # ground_truth_subplots(ground_truth)

    # Figure 2
    # ground_truth_xy(ground_truth)

    # Figure 3
    ground_truth_measurements(ground_truth, measurements)


def main():

    # Load the data from all files
    data = load_data(data_params['paths'], data_params['columns'])

    # plot_figures(data['ground_truth'], data['measurements'])

    # todo: delete the first element in the control and measurements
    kalman_filter(data['controls'], data['measurements'])


if __name__ == '__main__':
    main()
