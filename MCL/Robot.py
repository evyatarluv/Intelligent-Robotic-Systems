import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from .Ploter import config_plot


# Deterministic motion model of the robot
def x_motion(x, theta, u_1, u_2, noise_std):
    """
    The motion model of x pose
    :param x: float, current x of the robot
    :param theta: float, current theta of the robot
    :param u_1: float, turn command in radians
    :param u_2: float, movement command
    :param noise_std: float, std of the movement
    :return: float, new x pose
    """

    noise = np.random.normal(0, noise_std, 1)
    return x + u_2 * np.cos(theta + u_1) + noise


def y_motion(y, theta, u_1, u_2, noise_std):
    """
    The motion model of y pose
    :param y: float, current x of the robot
    :param theta: float, current theta of the robot
    :param u_1: float, turn command in radians
    :param u_2: float, movement command
    :param noise_std: float, std of the movement
    :return: float, new y pose
    """

    noise = np.random.normal(0, noise_std, 1)
    return y + u_2 * np.cos(theta + u_1) + noise


def theta_motion(theta, u_1, noise_std):
    """
    The motion model of theta pose
    :param theta: float, current theta of the robot
    :param u_1: float, turn command in radians
    :param noise_std: float, std of the turn
    :return: float, new theta pose
    """

    noise = np.random.normal(0, noise_std, 1)
    return theta + u_1


class Robot:
    """
    the robot class, we will use this to describe a robot
    """
    def __init__(self, world_size=100, init_pose=None):
        """
        creating a robot object
        :param init_pose: the init position of the robot
        :param world_size: the world size in pixels
        """

        self._world_size = world_size

        # Pose declaration
        if init_pose is None:
            self.x = np.random.rand() * self._world_size
            self.y = np.random.rand() * self._world_size
            self.theta = np.random.rand() * 2 * np.pi
        else:
            self.x, self.y, self.theta = init_pose

        # Noise declaration
        self.forward_noise = 0
        self.turn_noise = 0
        self.sense_distance_noise = 0
        self.sense_noise_range = 0
        self.sense_noise_bearing = 0

        # Motion Model
        self.x_motion = x_motion
        self.y_motion = y_motion
        self.theta_motion = theta_motion

    def __str__(self):
        """"
        printing the pose
        """

        return '[x = {}, y = {} heading = {}]'.format(round(self.x, 3), round(self.y, 3), round(self.theta, 3))

    def set(self, new_x, new_y, new_orientation):
        """
        setting the configuration of the robot
        :param new_x: the new x coordinate
        :param new_y: the new y coordinate
        :param new_orientation: the new orientation
        """
        if new_x < 0 or new_x >= self._world_size:
            raise Exception('X coordinate out of bound')

        if new_y < 0 or new_y >= self._world_size:
            raise Exception('Y coordinate out of bound')

        if new_orientation < 0.0 or new_orientation >= 2 * np.pi:
            raise Exception('Orientation must be in [0,2pi]')

        self.x = new_x
        self.y = new_y
        self.theta = new_orientation

    def plot(self, mycolor='b', style='robot', show=False, markersize=1, r=3):
        """
        plotting the pose of the robot in the world
        :param markersize:
        :param r: radius of the robot
        :param mycolor: the color of the robot
        :param style: the style to plot with
        :param show: if to show or not show - used to create a new figure or not
        """
        if style == 'robot':
            phi = np.linspace(0, 2 * np.pi, 101)
            # plot robot body
            plt.plot(self.x + r * np.cos(phi), self.y + r * np.sin(phi), color=mycolor)
            # plot heading direction
            plt.plot([self.x, self.x + r * np.cos(self.theta)], [self.y, self.y + r * np.sin(self.theta)], color=mycolor)

        elif style == 'particle':
            plt.plot(self.x, self.y, '.', color=mycolor, markersize=markersize)
        else:
            print('unknown style')

        if show:
            plt.show()

    def set_noise(self, new_forward_noise, new_turn_noise,new_sense_noise_range, new_sense_noise_bearing):
        """
        setting the noise if pose of the robot
        :param new_forward_noise: the noise for moving forward
        :param new_turn_noise: the noise in the turn of the robot
        :param new_sense_noise_range: the noise in range measurement
        :param new_sense_noise_bearing: the noise in bearing measurement
        """

        self.forward_noise = new_forward_noise
        self.turn_noise = new_turn_noise
        self.sense_noise_range = new_sense_noise_range
        self.sense_noise_bearing = new_sense_noise_bearing

    def get_pose(self):
        """
        returning the pose vector
        :return: (x, y, theta) the pose vector
        """
        return self.x, self.y, self.theta

    def move(self, u_1, u_2):
        """
        The method move the robot according to the given motor command - u1, u2
        :param u_1: float in range [0, 2 * pi), turn command
        :param u_2: float in range [0, inf), movement command
        :return:
        """

        # Compute the new pose of the robot
        new_x = self.x_motion(self.x, self.theta, u_1, u_2, self.forward_noise)
        new_y = self.y_motion(self.y, self.theta, u_1, u_2, self.forward_noise)
        new_theta = self.theta_motion(self.theta, u_1, self.turn_noise)

        # Set the new pose as the robot pose
        self.set(new_x, new_y, new_theta)

    def sense(self, landmarks):

        """
        The method measure distance and bearing to each landmark.
        Each measurement from the returns list contain tuples (range, bearing)
        :param landmarks: list with m tuples, each tuple contain (x, y) pose of landmark
        :return: list with m tuples, when m is the amount of landmarks
        """

        measurements = []

        for m in landmarks:

            # Compute range and bearing of the robot from the current landmark
            r = np.sqrt((m[0] - self.x) ** 2 + (m[1] - self.y) ** 2) + self.sense_noise_range
            phi = np.arctan2(m[1] - self.y, m[0] - self.x) - self.theta + self.sense_noise_bearing

            # Append the measurement
            measurements.append((r, phi))

        return measurements

