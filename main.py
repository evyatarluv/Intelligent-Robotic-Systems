import pandas as pd
import numpy as np
from load_data import load_data
from figures import *
from extended_kalman_filter import extended_kalman_filter
import matplotlib.pyplot as plt

# Parameters

ellipse_times = [int(i / 0.05) for i in [0, 2, 4, 6, 8, 10]]  # times to show an ellipse


def main():
    # Load the data from all files
    data = load_data()

    # First Run
    ekf_mu, ekf_sigma = extended_kalman_filter(data['controls'], data['measurements'])

    subplots(data['ground_truth'], ekf_mu)

    ax = xy_path(data['ground_truth'], ekf_mu, return_axes=True)

    add_confidence_ellipse(ax, ekf_sigma, ekf_mu, times=ellipse_times)

    # Second Run
    # Run EKF while' override default sigma_r params
    new_param = {'variances': {'r': 0.01, 'phi': 0.007}}
    ekf_mu, ekf_sigma = extended_kalman_filter(data['controls'], data['measurements'], new_param)

    subplots(data['ground_truth'], ekf_mu)

    ax = xy_path(data['ground_truth'], ekf_mu, return_axes=True)
    add_confidence_ellipse(ax, ekf_sigma, ekf_mu, times=ellipse_times)


if __name__ == '__main__':
    main()
