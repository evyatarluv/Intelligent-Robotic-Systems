import pandas as pd
import numpy as np
from figures import *
from extended_kalman_filter import extended_kalman_filter, h_function
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
    # ground_truth_subplots(ground_truth)

    # Figure 2
    # ground_truth_xy(ground_truth)

    # Figure 3
    ground_truth_measurements(ground_truth, measurements)


def main():

    # Load the data from all files
    data = load_data(data_params['paths'], data_params['columns'])

    # plot_figures(data['ground_truth'], data['measurements'])

    kf = extended_kalman_filter(data['controls'], data['measurements'])
    figure_2(data['ground_truth'], None, data['measurements'])

    # Debug
    x = data['ground_truth'].x
    y = data['ground_truth'].y
    theta = data['ground_truth'].theta
    t = data['ground_truth'].time
    kf_x = kf[:, 0]
    kf_y = kf[:, 1]
    kf_theta = kf[:, 2]
    r_meas = data['measurements'].r
    phi_meas = data['measurements'].phi

    # ground truth r, phi
    gt = np.array([h_function(x[i], y[i], theta[i]) for i in range(len(x))])
    r = gt[:, 0]
    phi = gt[:, 1]

    # scatter
    # plt.scatter(t, theta, marker='x')
    # plt.scatter(t[1:], kf_theta, marker='x')
    # plt.show()

    # plot
    # plt.plot(t, x)
    # plt.plot(t[1:], kf_x)
    # plt.show()

    # plt.plot(t, y)
    # plt.plot(t[1:], kf_y)
    # plt.show()

    # plt.plot(t, theta)
    # plt.plot(t[1:], kf_theta)
    # plt.show()

    # plt.plot(x, y)
    # plt.plot(kf_x, kf_y)
    # plt.show()

    # plt.plot(t, phi_meas)
    # plt.plot(t, phi)
    # plt.show()

    # plt.plot(t, data['controls'].omega)
    # plt.show()


if __name__ == '__main__':
    main()
