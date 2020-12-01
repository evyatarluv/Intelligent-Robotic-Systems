""""
This script include the implementation of kalman filter.
The main function is `kalman_filter` while all the other are auxiliary functions.

Mu vector define as [x, y, theta], control vector define as [v, omega] and measurement
vector define as [r, phi]. The landmark parameter is the location of the landmark as [x, y].

For the matrix which change in each time step there is a unique function to construct it.
The function construct the matrix given the current state or control. The function named `construct_X`
while X is the name of the matrix.
"""

import numpy as np

# Kalman filter parameters
kf_params = {
    'A': np.identity(3),
    'variances': {'r': 4, 'phi': 0.07},
    'mu_0': np.array([0, 0, 0]),
    'sigma_0': np.array([[0.5, 0, 0],
                         [0, 0.5, 0],
                         [0, 0, 100]]),
    'landmark': [8, 9],
    'dt': 0.05,
    'alpha': [0.05, 0.05, 0.05, 0.05],
}


def construct_B(theta, omega):
    """
    This function construct the current B matrix - how the control change the state.
    :param theta: current theta state
    :param omega: omega control
    :return: ndarray of the B matrix
    """

    dt = kf_params['dt']

    B = np.array([[np.cos(theta + omega * dt), 0],
                  [0, np.sin(theta + omega * dt)],
                  [0, dt]])

    return B


def construct_R(v, omega):
    """
    Construct the R matrix, the matrix describe the noise of the motion.
    :param v: current velocity control
    :param omega: current omega control
    :return: ndarray of the R matrix
    """

    alpha = kf_params['alpha']

    # Calculate the variance of v & omega
    sigma_v = alpha[0] * (v ** 2) + alpha[1] * (omega ** 2)
    sigma_omega = alpha[2] * (v ** 2) + alpha[3] * (omega ** 2)

    return np.diag([sigma_v, sigma_omega])


def construct_H(x, y):
    """
    This function construct H function which map the state to observation.
    :return:
    """

    m_x = kf_params['landmark'][0]  # landmark x pos
    m_y = kf_params['landmark'][1]  # landmark y pos
    q = (m_x - x) ** 2 + (m_y - y) ** 2  # auxiliary variable

    H = np.array([[(m_x - x) / np.sqrt(q), (m_y - y) / np.sqrt(q), 0],
                  [(y - m_y) / np.sqrt(q), (m_x - x) / np.sqrt(q)]])

    return H


def predict(mu, sigma, control):
    """
    The prediction step in the Kalman filter algorithm.
    Use the current mu & sigma in order to predict the next mu & sigma
    :param mu: list of state before the control
    :param sigma: ndarray of the the sigma matrix
    :param control: list of the control the robot did
    :return: predicted mu and sigma of the next step
    """

    # Get the relevant matrices
    A = kf_params['A']
    R = construct_R(v=control[0], omega=control[1])
    B = construct_B(theta=mu[2], omega=control[1])

    # Calculate the next mu and sigma
    new_mu = A @ mu + B @ control
    new_sigma = A @ sigma @ A.T + R

    return new_mu, new_sigma


def kalman_gain(mu_bar, sigma_bar):
    """
    This function calculate the Kalman gain.
    :param mu_bar: prediction of the current state
    :param sigma_bar: prediction of the current sigma matrix
    :return: list with the kalman gain for each measurement
    """

    # Get relevant matrices
    Q = np.diag([kf_params['variances']['r'], kf_params['variances']['phi']])
    H = construct_H(x=mu_bar[0], y=mu_bar[1])

    # Auxiliary calculation for Kalman gain
    right_phrase = np.linalg.inv(H @ sigma_bar @ H.T + Q)
    left_phrase = sigma_bar @ H.T

    # Return Kalman gain
    return right_phrase @ left_phrase


def kalman_filter(control, measurement):
    """
    Main function in the implementation of kalman filter.
    Get the control and measurements of the robot and return the localization according to kalman filter.
    :param control: list with the control of the robot
    :param measurement: list with the measurements of the robot
    :return: list with the localization of the robot
    """

    steps = len(control)  # num of steps
    localization = []  # output list
    mu = kf_params['mu_0']  # init mu
    sigma = kf_params['sigma_0']  # init sigma

    # For each robot step
    for i in range(steps):
        
        mu_bar, sigma_bar = predict(mu, sigma, control.iloc[i].to_list())

        K = kalman_gain(mu_bar, sigma_bar)

        # mu, sigma = update_measure()

        localization.append(mu)
