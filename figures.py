"""
This script include all the plot functions.
Each function responsible for different type of plot which was requested in the exercise.
"""

import matplotlib.pyplot as plt
import numpy as np


def ground_truth_subplots(ground_truth):

    # Extract the data from the data frame
    time = ground_truth['time']
    x = ground_truth['x']
    y = ground_truth['y']
    theta = ground_truth['theta']

    # Figure 1:  ground truth
    fig, ax = plt.subplots(3, sharex='col')
    ax[0].set_title('Figure 1: Ground Truth')

    ax[0].plot(time, x, color='black')
    ax[1].plot(time, y, color='black')
    ax[2].plot(time, theta, color='black')

    ax[0].set(ylabel='X (m)')
    ax[1].set(ylabel='Y (m)')
    ax[2].set(ylabel='Theta (rad)', xlabel='Time (s)')

    plt.show()


def ground_truth_xy(ground_truth):

    # Extract the data from the data frame
    time = ground_truth['time']
    x = ground_truth['x']
    y = ground_truth['y']

    plt.plot(x, y, color='black')
    plt.title('Figure 2: XY-Path')
    plt.show()


def ground_truth_measurements(ground_truth, measurements):

    # Extract data
    x_truth = ground_truth['x']
    y_truth = ground_truth['y']
    theta = ground_truth['theta']
    r = measurements['r']
    phi = measurements['phi']
    landmark_x = 8
    landmark_y = 9

    # Convert from [r, theta] to [x, y] coordinates
    x_meas = [r[i] * np.cos(np.pi + phi[i] + theta[i]) + landmark_x for i in range(len(r))]
    y_meas = [r[i] * np.sin(np.pi + phi[i] + theta[i]) + landmark_y for i in range(len(r))]

    # Plot
    plt.plot(x_truth, y_truth, color='black', label='Ground Truth')
    plt.scatter(x_meas, y_meas, marker='+', color='blue', label='Measurements')
    plt.scatter(landmark_x, landmark_y, marker='o', color='red', linewidths=2, label='Landmark')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XY Path')
    plt.legend()
    plt.show()

