""""
This script include the implementation of kalman filter.
The main function is `kalman_filter` while all the other are auxiliary functions.

Definitions:
mu: [x, y, theta]
control: [v, omega]
measurement: [r, phi]
landmark: [x, y]

For the matrix which change in each time step there is a unique function to construct it.
The function construct the matrix given the current state or control. The function named `construct_X`
while X is the name of the matrix.
"""

import numpy as np

# Kalman filter parameters
kf_params = {
    'variances': {'r': 4, 'phi': 0.07},
    'mu_0': np.array([0, 0, 0]),
    'sigma_0': np.array([[0.5, 0, 0],
                         [0, 0.5, 0],
                         [0, 0, 100]]),
    'landmark': [8, 9],
    'dt': 0.05,
    'alpha': [0.01, 0.01, 0.01, 0.01],
}


def preprocess_kf_data(controls, measurements):
    """
    Pre-process the data in order to fit the kalman filter function.
    The function:
        1. Convert to numpy array
        2. Drop the time column
        3. Drop the first row which is not needed.
    :param controls: data frame of the controls of the robot
    :param measurements: data frame of  the measurements of the robot
    :return:
    """

    # Convert to ndarray
    controls = controls.to_numpy()
    measurements = measurements.to_numpy()

    # Drop the time column
    controls = np.delete(controls, obj=0, axis=1)
    measurements = np.delete(measurements, obj=0, axis=1)

    # Drop the first row
    controls = np.delete(controls, obj=0, axis=0)
    measurements = np.delete(measurements, obj=0, axis=0)

    return controls, measurements


def construct_B(theta, omega):
    """
    This function construct the current B matrix - how the control change the state.
    :param theta: current theta state
    :param omega: omega control
    :return: ndarray of the B matrix
    """

    dt = kf_params['dt']

    B = np.array([[np.cos(theta + omega * dt), 0],
                  [np.sin(theta + omega * dt), 0],
                  [0, dt]])

    return B


def construct_R(theta, v, omega):
    """
    Construct the R matrix, the matrix describe the noise of the motion.
    To get the R matrix the function use the formula: R = V @ M @ V.T.
    V is the derivative of the dynamic by the control and M is cov matrix of the control.
    :param theta: current theta of the robot
    :param v: current velocity control
    :param omega: current omega control
    :return: ndarray of the R matrix
    """

    dt = kf_params['dt']
    alpha = kf_params['alpha']

    # Construct V & M matrices
    V = np.array([[np.cos(theta + omega * dt), -v * dt * np.sin(theta + omega * dt)],
                  [np.sin(theta + omega * dt), v * dt * np.cos(theta + omega * dt)],
                  [0, dt]])
    M = np.diag([alpha[0] * (v ** 2) + alpha[1] * (omega ** 2),
                 alpha[2] * (v ** 2) + alpha[3] * (omega ** 2)])

    # Construct R matrix by the above-mentioned formula
    R = V @ M @ V.T

    return R


def construct_H(x, y):
    """
    This function construct Jacobians H matrix.

    :param x: x pos of the robot
    :param y: y pos of the robot
    :return: ndarray H matrix
    """

    m_x = kf_params['landmark'][0]  # landmark x pos
    m_y = kf_params['landmark'][1]  # landmark y pos
    q = (m_x - x) ** 2 + (m_y - y) ** 2  # auxiliary variable

    H = np.array([[(m_x - x) / np.sqrt(q), (y - m_y) / np.sqrt(q), 0],
                  [(m_y - y) / q, (m_x - x) / q, -1]])

    return H


def construct_G(theta, v, omega):
    """
    This function construct the G matrix
    :param theta: theta ori of the robot
    :param v: velocity control
    :param omega: omega control
    :return: ndarray G matrix
    """

    dt = kf_params['dt']

    G = np.array([[1, 0, -v * np.sin(theta + omega * dt)],
                  [0, 1, v * np.cos(theta + omega * dt)],
                  [0, 0, 1]])

    return G


def h_function(x, y, theta):
    """
    This function map the state of the robot into an observation.
    The function represent the non-linear h(t) function in the EKF algorithm.
    h(t): [x, y, theta] -> [r, phi]
    :param x: x pos of the robot
    :param y: y pos of the robot
    :param theta: theta orientation of the robot
    :return: ndarray of the observation
    """

    # Landmark x & y position
    m_x = kf_params['landmark'][0]
    m_y = kf_params['landmark'][1]

    # Calculate the observation
    r = np.sqrt((m_x - x) ** 2 + (m_y - y) ** 2)
    phi = np.arctan2(m_y - y, m_x - x) - theta

    return np.array([r, phi])


def g_function(state, control):
    """
    This function represent the g non-linear function which describe the state transition.
    :param state: (t-1) state vector
    :param control: current control vector
    :return: new state vector
    """

    dt = kf_params['dt']
    x, y, theta = state[0], state[1], state[2]
    v, omega = control[0], control[1]

    new_x = x + v * np.cos(theta + omega * dt)
    new_y = y + v * np.sin(theta + omega * dt)
    new_theta = theta + omega * dt

    return np.array([new_x, new_y, new_theta])


def predict(previous_mu, previous_sigma, control):
    """
    The prediction step in the Kalman filter algorithm.
    Use the current mu & sigma in order to predict the next mu & sigma
    :param previous_mu: list of state before the control
    :param previous_sigma: ndarray of the the sigma matrix
    :param control: list of the control the robot did
    :return: predicted mu and sigma of the next step
    """

    # Calculate mu_bar
    mu_bar = g_function(state=previous_mu, control=control)

    # Calculate sigma_bar
    R = construct_R(theta=previous_mu[2], v=control[0], omega=control[1])
    G = construct_G(theta=previous_mu[2], v=control[0], omega=control[1])

    sigma_bar = G @ previous_sigma @ G.T + R

    return mu_bar, sigma_bar


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
    # right_phrase = np.linalg.inv(H @ sigma_bar @ H.T + Q)
    # left_phrase = sigma_bar @ H.T

    k = sigma_bar @ H.T @ np.linalg.inv(H @ sigma_bar @ H.T + Q)

    return k


def update_measure(mu_bar, sigma_bar, K, measurement):
    """
    Measurement update step in the Kalman filter algorithm.
    The function calculate the new mu & sigma of the robot.
    :param measurement: the current measurement (z) of the robot.
    :param mu_bar: ndarray of the predicted mu
    :param sigma_bar: ndarray of the predicted sigma
    :param K: Kalman gain
    :return: calculated mu & sigma
    """

    # Get H matrix
    H = construct_H(x=mu_bar[0], y=mu_bar[1])

    # Calculate mu
    h_mu = h_function(x=mu_bar[0], y=mu_bar[1], theta=mu_bar[2])
    mu = mu_bar + K @ (measurement - h_mu)

    # Calculate sigma
    KH = K @ H
    sigma = (np.identity(len(KH)) - KH) @ sigma_bar

    return mu, sigma


def extended_kalman_filter(control, measurement):
    """
    Main function in the implementation of the Kalman filter.
    Get the control and measurements of the robot and return the localization according to kalman filter.
    :param control: array with the control of the robot
    :param measurement: array with the measurements of the robot
    :return: list with the localization of the robot
    """

    # Init params
    localization = []
    mu = kf_params['mu_0']
    sigma = kf_params['sigma_0']
    control, measurement = preprocess_kf_data(control, measurement)

    # For each robot step
    for i in range((len(control))):

        # Prediction
        mu_bar, sigma_bar = predict(mu, sigma, control[i])

        # Kalman gain
        K = kalman_gain(mu_bar, sigma_bar)

        # Measurement update
        mu, sigma = update_measure(mu_bar, sigma_bar, K, measurement[i])

        # Append the current mu
        localization.append(mu)

    return np.array(localization)
