from copy import deepcopy
import numpy as np
from MCL.Robot import Robot
from MCL.World import World


class MCL:

    """
    A class for the implementation of MCL algorithm.

    Attributes:
        particles (list of Robot): The particles in the algorithm where each particle is a Robot, with length m.
        landmarks (list of tuple): List with each landmark position as (x,y) tuple.
    """
    def __init__(self, m, landmarks, robot):
        """
        Init MCL object
        :param m: int, amount of particles
        :param landmarks: list, contains the landmarks positions
        :param robot: Robot, the robot we applied the MCL algorithm on it
        """

        self.landmarks = deepcopy(landmarks)
        self.particles = [deepcopy(robot) for i in range(m)]

    def localize(self, motion_commands, measurements):
        """
        This method localize the robot using MCL algorithm.
        The method get the current motion commands u and the robot measurements (known as z).
        The method uses the previous step particles to compute current step's list of particles. The mean
        of the current step's resampled particles is the belief for the robot's position.
        :param measurements: list, measurements of the robot to the landmarks.
        :param motion_commands: tuple, motion commands of the robot.
        :return: list, resampled particles where each particle is a tuple of (x,y).
        """

        u_1, u_2 = motion_commands

        # 1. Move particles
        self.particles = [p.move(u_1, u_2) for p in self.particles]

        # 2. Compute weights for particles
        weights = self.compute_weights(measurements)

        # 3. Resample particles
        resampled = self.resample_particles(weights)

        return resampled

    def compute_weights(self, measurements):

        """
        The method compute weight for each particle according to the particle's measurements.
        :param measurements: list, measurements of the robot regarding the landmarks
        :return: list, weight for each particle
        """

        weights = []

        # For each particle
        for p in self.particles:

            # Sense the landmarks and compute probabilities
            particle_measurements = p.sense(self.landmarks)
            probabilities = []

            # For each landmark measure
            for landmark_index, measure in enumerate(particle_measurements):

                landmark_prob = p.measurement_probability(pose=p.get_pose(),
                                                          measurement=measurements[landmark_index],
                                                          landmark=self.landmarks[landmark_index])
                probabilities.append(landmark_prob)

            # Append the current particle weight
            weights.append(np.mean(probabilities))

        return weights





