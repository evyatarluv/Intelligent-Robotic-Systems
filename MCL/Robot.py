import copy
import numpy as np
from matplotlib import pyplot as plt
from .Ploter import config_plot


def norm_pdf(x, mean, sd):
    """
    Probability density function at x of the given (mu, sigma) normal distribution args.
    :param x: float, value
    :param mean: float, mu
    :param sd: float, sigma
    :return: probability between 0 to 1
    """

    var = float(sd)**2
    denom = (2 * np.pi * var) ** .5
    num = np.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


class Robot:
    """
    the robot class, we will use this to describe a robot
    """
    def __init__(self, world_size=100, init_pose=None, noise_std=None):
        """
        creating a robot object
        :param init_pose: tuple, the init position of the robot
        :param noise_std: dict, noise of the robot
        :param world_size: int, the world size in pixels
        """

        self._world_size = world_size

        # Init pose
        if init_pose is None:
            self.x = np.random.rand() * self._world_size
            self.y = np.random.rand() * self._world_size
            self.theta = np.random.rand() * 2 * np.pi
        else:
            self.x, self.y, self.theta = init_pose

        # Noise
        if noise_std is None:
            self.noise_std = {'forward': 0, 'turn': 0, 'range': 0, 'bearing': 0}
        else:
            self.noise_std = copy.deepcopy(noise_std)

        # Path
        self.path = [(self.x, self.y)]

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

        # I prefer not raising error I can handle myself
        self.x = new_x % self._world_size
        self.y = new_y % self._world_size
        self.theta = new_orientation

    def plot(self, mycolor='b', style='robot', show=False, markersize=1, r=3):
        """
        plotting the pose of the robot in the world
        :param markersize:
        :param r: float, radius of the robot
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

    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise_range, new_sense_noise_bearing):
        """
        setting the noise if pose of the robot
        :param new_forward_noise: the noise for moving forward
        :param new_turn_noise: the noise in the turn of the robot
        :param new_sense_noise_range: the noise in range measurement
        :param new_sense_noise_bearing: the noise in bearing measurement
        """

        self.noise_std['forward'] = new_forward_noise
        self.noise_std['turn'] = new_turn_noise
        self.noise_std['range'] = new_sense_noise_range
        self.noise_std['bearing'] = new_sense_noise_bearing

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

        # Add noise to the motion commands
        u_1 += np.random.normal(0, self.noise_std['turn'])
        u_2 += np.random.normal(0, self.noise_std['forward'])

        # Compute the new pose of the robot according the motion model
        new_x = self.x + u_2 * np.cos(self.theta + u_1)
        new_y = self.y + u_2 * np.sin(self.theta + u_1)
        new_theta = self.theta + u_1

        # Set the new pose as the robot pose
        self.set(new_x, new_y, new_theta)

        # Append the new pose to the path
        self.path.append((self.x, self.y))

    def sense(self, landmarks):

        """
        The method measure distance and bearing to each landmark.
        Each measurement from the returns list contain tuples (range, bearing)
        :param landmarks: list with m tuples, each tuple contain (x, y) pose of landmark
        :return: list with m tuples, when m is the amount of landmarks
        """

        measurements = []

        for m in landmarks:

            # Get measurement noise
            range_noise = np.random.normal(0, self.noise_std['range'])
            bearing_noise = np.random.normal(0, self.noise_std['bearing'])

            # Compute range and bearing of the robot from the current landmark
            r = np.sqrt((m[0] - self.x) ** 2 + (m[1] - self.y) ** 2) + range_noise
            phi = np.arctan2(m[1] - self.y, m[0] - self.x) - self.theta + bearing_noise

            # Append the measurement
            measurements.append((r, phi))

        return measurements

    def measurement_probability(self, measurement, landmark):
        """
        The method compute the probability for a given measurement to be observed when being
        in a given pose.
        :param landmark: landmark position, (x,y)
        :param measurement: measurement the robot measure, (range, bearing)
        :return: float, probability between 0 to 1
        """

        # Extract values
        x, y, theta = self.get_pose()
        r, phi = measurement
        m_x, m_y = landmark

        # Compute measurement giving the pose
        meas_r = np.sqrt((m_x - x) ** 2 + (m_y - y) ** 2)
        meas_phi = np.arctan2(m_y - y, m_x - x) - theta

        # Compute the probability for each measure
        prob_r = norm_pdf(r - meas_r, 0, self.noise_std['range'])
        prob_phi = norm_pdf(phi - meas_phi, 0, self.noise_std['bearing'])

        return prob_r * prob_phi

