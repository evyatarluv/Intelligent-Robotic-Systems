"""
This script include all the plot functions.
Each function responsible for different type of plot which was requested in the exercise.
"""

import matplotlib.pyplot as plt
import numpy as np


def figure_1(ground_truth, estimation):
    """
    Generate requested `Figure 1` - subplots of [x, y, theta].
    :param ground_truth: ground-truth of the robot
    :param estimation: localization estimation by algorithm
    :return:
    """

    # Extract the data from the data frame
    time = ground_truth['time']
    x = ground_truth['x']
    y = ground_truth['y']
    theta = ground_truth['theta']

    # Set the figure
    fig, ax = plt.subplots(3, sharex='col')
    ax[0].set_title('Figure 1: [x, y, theta]')

    ax[0].plot(time, x, color='black')
    ax[1].plot(time, y, color='black')
    ax[2].plot(time, theta, color='black', label='Ground Truth')

    if estimation is not None:
        ax[0].plot(time[1:], estimation[:, 0], color='red')
        ax[1].plot(time[1:], estimation[:, 1], color='red')
        ax[2].plot(time[1:], estimation[:, 2], color='red', label='EKF')

    # Set labels
    ax[0].set(ylabel='X (m)')
    ax[1].set(ylabel='Y (m)')
    ax[2].set(ylabel='Theta (rad)', xlabel='Time (s)')

    # Add legend
    ax[2].legend(loc='lower right', bbox_to_anchor=(1, 0))

    plt.show()


def figure_2(ground_truth, estimation, measurements):
    """
    Figure 2 - XY path of the robot
    :param measurements: ndarray of the measurements, None if nothing
    :param ground_truth: ndarray of the ground-truth of the robot
    :param estimation: ndarray of the estimation by an algorithm, None if not estimation currently
    :return:
    """

    # Extract the data from the data frame
    x = ground_truth['x']
    y = ground_truth['y']
    theta = ground_truth['theta']

    # Plot the ground-truth
    plt.plot(x, y, color='black', label='Ground Truth')

    # Plot the estimation
    if estimation is not None:
        plt.plot(estimation[0], estimation[1], color='red')

    # Plot the measurements
    if measurements is not None:

        r = measurements['r']
        phi = measurements['phi']
        landmark_x = 8
        landmark_y = 9

        # Convert from [r, theta] to [x, y] coordinates
        x_meas = [r[i] * np.cos(np.pi + phi[i] + theta[i]) + landmark_x for i in range(len(r))]
        y_meas = [r[i] * np.sin(np.pi + phi[i] + theta[i]) + landmark_y for i in range(len(r))]

        # Plot
        plt.scatter(x_meas, y_meas, marker='+', color='blue', label='Measurements')
        plt.scatter(landmark_x, landmark_y, marker='o', color='red', linewidths=3, label='Landmark')

    # Some aesthetics to get and handsome graph
    plt.title('Figure 2 - XY Path')
    plt.legend()

    plt.show()

