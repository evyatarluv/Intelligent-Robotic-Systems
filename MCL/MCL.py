from copy import deepcopy
import numpy as np
from tqdm import tqdm


class MCL:

    """
    A class for the implementation of MCL algorithm.

    Attributes:
        particles (list of Robot): The particles in the algorithm where each particle is a Robot, with length m.
        landmarks (list of tuple): List with each landmark position as (x,y) tuple.
        location (tuple): The current estimated location of the robot as (x,y) tuple.
    """
    def __init__(self, robot, landmarks, m):
        """
        Init MCL object
        :param m: int, amount of particles
        :param landmarks: list, contains the landmarks positions
        :param robot: Robot, the robot we applied the MCL algorithm on it
        """

        self.landmarks = deepcopy(landmarks)
        self.particles = [deepcopy(robot) for i in range(m)]
        self.location = self.estimate_location()

    def localize(self, motion_commands, measurements):
        """
        This method localize the robot using MCL algorithm.
        The method get the current motion commands u and the robot measurements (known as z).
        The method uses the previous step particles to compute current step's list of particles. The mean
        of the current step's resampled particles is the belief for the robot's position.
        :param measurements: list, measurements of the robot to the landmarks.
        :param motion_commands: tuple, motion commands of the robot.
        :return: lists of Robot, the sample list and the resample list
        """

        u_1, u_2 = motion_commands

        # Move particles
        [p.move(u_1, u_2) for p in self.particles]
        sample = deepcopy(self.particles)

        # Compute weights for particles
        weights = self.compute_weights(measurements)

        # Resample particles
        self.particles = self.resample(weights)

        # Update estimated location
        self.location = self.estimate_location()

        return sample, self.particles

    def compute_weights(self, measurements):

        """
        The method compute weight for each particle according to the particle's measurements.
        :param measurements: list, measurements of the robot regarding the landmarks
        :return: list, weight for each particle
        """

        weights = []

        # For each particle
        for p in tqdm(self.particles):

            # Sense the landmarks and compute probabilities
            particle_measurements = p.sense(self.landmarks)
            probabilities = []

            # For each landmark measure
            for landmark_index, measure in enumerate(particle_measurements):

                landmark_prob = p.measurement_probability(measurement=measurements[landmark_index],
                                                          landmark=self.landmarks[landmark_index])
                probabilities.append(landmark_prob)

            # Append the current particle weight
            weights.append(np.mean(probabilities))

        return weights

    def resample(self, weights):
        """
        The method resample particles from the current particles according to a given weights list
        :param weights: list, weight for each particle
        :return: list of Robot, the resampled particle
        """

        # Convert weights to probabilities
        weights_sum = np.sum(weights)
        prob = [w / weights_sum for w in weights]

        # Resample particles
        resample = np.random.choice(self.particles, size=len(self.particles), p=prob, replace=True)

        return resample

    def estimate_location(self):
        """
        The method estimate the robot location according the current particles.
        Currently, the estimation is about (x,y) position
        :return: tuple, the estimated location of the robot
        """

        positions = [(p.x, p.y) for p in self.particles]

        return tuple(np.mean(positions, axis=0))


