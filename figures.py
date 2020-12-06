"""
This script include all the plot functions.
Each function responsible for different type of plot which was requested in the exercise.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np

landmark = [8, 9]


def measurements_to_xy(measurements, theta):
    r = measurements['r']
    phi = measurements['phi']
    m_x = landmark[0]
    m_y = landmark[1]

    # Convert from [r, theta] to [x, y] coordinates
    x_meas = [r[i] * np.cos(np.pi + phi[i] + theta[i]) + m_x for i in range(len(r))]
    y_meas = [r[i] * np.sin(np.pi + phi[i] + theta[i]) + m_y for i in range(len(r))]

    return x_meas, y_meas


def subplots(ground_truth, estimation=None, measurements=None):
    """
    Generate requested `Figure 1` - subplots of [x, y, theta].
    :param measurements:
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

    ax[0].plot(time, x, color='black', label='Ground Truth')
    ax[1].plot(time, y, color='black')
    ax[2].plot(time, theta, color='black')

    # Add measurements
    if measurements is not None:
        meas_x, meas_y = measurements_to_xy(measurements, theta)
        meas_theta = [np.arctan2(landmark[1] - y[i], landmark[0] - x[i]) - measurements.phi[i]
                      for i in range(len(measurements.phi))]

        ax[0].scatter(time, meas_x, marker='+', color='blue', label='Measurements')
        ax[1].scatter(time, meas_y, marker='+', color='blue')
        ax[2].scatter(time, meas_theta, marker='+', color='blue')

    # Add estimations
    if estimation is not None:
        ax[0].plot(time[1:], estimation[:, 0], color='red', label='EKF')
        ax[1].plot(time[1:], estimation[:, 1], color='red')
        ax[2].plot(time[1:], estimation[:, 2], color='red')

    # Set labels
    ax[0].set(ylabel='X (m)')
    ax[1].set(ylabel='Y (m)')
    ax[2].set(ylabel='Theta (rad)', xlabel='Time (s)')

    # Add legend
    ax[0].legend(loc='lower right', bbox_to_anchor=(1, 0.8))

    plt.show()


def xy_path(ground_truth, estimation=None, measurements=None, return_axes=False):
    """
    Figure 2 - XY path of the robot
    :param return_axes: bool. If to return an axes object or to plot it
    :param measurements: ndarray of the measurements, None if nothing
    :param ground_truth: ndarray of the ground-truth of the robot
    :param estimation: ndarray of the estimation by an algorithm, None if not estimation currently
    :return:
    """

    # Extract the data from the data frame
    x = ground_truth['x']
    y = ground_truth['y']
    theta = ground_truth['theta']

    f, ax = plt.subplots()

    # Plot the ground-truth
    ax.plot(x, y, color='black', label='Ground Truth')

    # Plot the estimation
    if estimation is not None:
        ax.plot(estimation[:, 0], estimation[:, 1], color='red', label='EKF')

    # Plot the measurements
    if measurements is not None:
        m_x, m_y = landmark[0], landmark[1]
        x_meas, y_meas = measurements_to_xy(measurements, theta)

        # Plot
        plt.scatter(x_meas, y_meas, marker='+', color='blue', label='Measurements')
        plt.scatter(m_x, m_y, marker='o', color='red', linewidths=3, label='Landmark')

    # Some aesthetics to get and handsome graph
    ax.set_title('Figure 2 - XY Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    if return_axes:
        return ax
    else:
        plt.show()


def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)


def add_confidence_ellipse(ax, estimated_sigma, estimated_mean, times):

    # Add init mu and sigma
    init_sigma = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 100]])
    init_mu = np.array([0, 0, 0])
    estimated_sigma = np.insert(estimated_sigma, 0, init_sigma, 0)
    estimated_mean = np.insert(estimated_mean, 0, init_mu, 0)

    # Get x & y
    x = estimated_mean[times, 0]
    y = estimated_mean[times, 1]

    for i, t in enumerate(times):

        # Get x,y
        center_x, center_y = x[i], y[i]

        # Take the relevant cov matrix
        sigma = estimated_sigma[t, :, :]

        # Change it to contain only (x, y) data
        sigma = sigma[0: 2, 0: 2]

        # Add ellipse
        ellipse = get_cov_ellipse(sigma, (center_x, center_y), 2, fc='red', alpha=0.2)
        ax.add_artist(ellipse)
        ax.scatter(center_x, center_y, color='red', s=8.5)

    plt.show()
