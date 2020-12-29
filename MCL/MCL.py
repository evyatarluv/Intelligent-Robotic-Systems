from copy import deepcopy
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from .Robot import Robot
import matplotlib.pyplot as plt


class MCL:
    """
    A class for the implementation of MCL algorithm.

    Attributes:
        m (int): The amount of particles.
        particles (list of Robot): The particles in the algorithm where each particle is a Robot, with length m.
        landmarks (list of tuple): List with each landmark position as (x,y) tuple.
        particles (list of Robot): The (resampled) particles of the current step.
        path (list of tuple): The estimated path of the robot computed by MCL.
        estimated_robot (Robot): The estimated Robot the algorithm computed to the current step, represent
                                    where the algorithm believe the robot's location.
    """

    def __init__(self, robot, landmarks, m):
        """
        Init MCL object
        :param m: int, amount of particles
        :param landmarks: list, contains the landmarks positions
        :param robot: Robot, the robot we applied the MCL algorithm on it
        """

        self.landmarks = deepcopy(landmarks)
        self.particles = []
        self.estimated_robot = deepcopy(robot)
        self.m = m
        self.path = [robot.get_pose()[:2]]

    def localize(self, motion_commands, measurements, plot=True):
        """
        This method localize the robot using MCL algorithm.
        The method get the current motion commands (u_t) and the robot measurements (z_t).
        The method uses the previous step computed location to compute current step's list of particles. The mean
        of the current step's resampled particles is the belief for the robot's position.
        :param measurements: list, measurements of the robot to the landmarks.
        :param motion_commands: tuple, motion commands of the robot.
        :param plot: bool, indicate if to plot the sampled and resampled particles
        :return:
        """

        u_1, u_2 = motion_commands
        weights = []
        self.particles = []

        # For each particle
        for i in range(self.m):

            # Create particle and move it
            particle = deepcopy(self.estimated_robot)
            particle.move(u_1, u_2)

            # Compute weight
            weight = self.compute_weight(measurements, particle)

            # Append particle and weight
            self.particles.append(particle)
            weights.append(weight)

        # Resample
        self.resample(weights, plot)

        # Update estimated location
        self.update_estimation()

    def compute_weight(self, measurements, particle):
        """
        The method get a list of measurements the real robot sense and a particle.
        Then the method compute the weight whether the particle is consist with the measurements vector.
        :param measurements: list, the measurements the real robot sense.
        :param particle: Robot, the current particle we need to compute weight to it.
        :return: float, weight of the particle
        """

        # Sense the landmarks and compute probabilities
        particle_measurements = particle.sense(self.landmarks)
        prob = []

        # For each landmark measure
        for landmark_index, measure in enumerate(particle_measurements):

            # Compute the probability & append it
            landmark_prob = particle.measurement_probability(measurement=measurements[landmark_index],
                                                             landmark=self.landmarks[landmark_index])
            prob.append(landmark_prob)

        return np.prod(prob)

    def resample(self, weights, plot=True):
        """
        The method resample particles from the attribute particles and update the attribute.
        The resampling using a probabilities vector which calculated using the weights vector.
        :param weights: list, weight for each particle (size m)
        :param plot: bool, indicate if to plot the particles before and after the resampling step
        :return:
        """
        # Convert weights to probabilities
        weights_sum = np.sum(weights)
        prob = [w / weights_sum for w in weights]

        # Plot particles
        if plot:
            for p in self.particles:
                p.plot(style='particle', mycolor='black', markersize=2)

        # Resample particles
        resample_idx = set(np.random.choice(a=len(self.particles), size=len(self.particles),
                                            p=prob, replace=True))
        self.particles = [self.particles[i] for i in resample_idx]

        # Plot resampled particles
        if plot:
            for p in self.particles:
                p.plot(style='particle', mycolor='lightgrey', markersize=2)

    def update_estimation(self):
        """
        The method update the estimated_robot attribute which indicate the estimated location of the robot.
        :return:
        """
        # todo: estimate the location using average according the particle weight
        # Compute the estimated location using mean of the particles
        estimated_position = np.mean([p.get_pose() for p in self.particles], axis=0)
        x, y, theta = estimated_position

        # Update the estimated robot
        self.estimated_robot.set(x, y, theta)

        # Append to the path
        self.path.append((x, y))

    def get_estimated_location(self):
        """
        Get the estimated location of the robot
        :return: tuple, estimated (x, y, theta) of the robot
        """

        return self.estimated_robot.get_pose()
