import pandas as pd
import numpy as np
from figures import ground_truth_xy, ground_truth_subplots, ground_truth_measurements
from kalman_filter import kalman_filter
import matplotlib.pyplot as plt

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
    ground_truth_subplots(ground_truth)

    # Figure 2
    # ground_truth_xy(ground_truth)

    # Figure 3
    ground_truth_measurements(ground_truth, measurements)


def main():

    # Load the data from all files
    data = load_data(data_params['paths'], data_params['columns'])

    # plot_figures(data['ground_truth'], data['measurements'])

    kf = kalman_filter(data['controls'], data['measurements'])

    # Debug - plot
    x = data['ground_truth'].x
    y = data['ground_truth'].y
    kf_x = kf[:, 0]
    kf_y = kf[:, 1]
    kf_theta = kf[:, 2]
    plt.plot(x, y)
    plt.plot(kf_x, kf_y)
    plt.show()

    print(kf.shape)


if __name__ == '__main__':
    main()
