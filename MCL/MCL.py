from copy import deepcopy
import numpy as np
from tqdm import tqdm
from .Robot import Robot
import matplotlib.pyplot as plt


class MCL:
    """
    A class for the implementation of MCL algorithm.

    Attributes:
        particles (list of Robot): The particles in the algorithm where each particle is a Robot, with length m.
        landmarks (list of tuple): List with each landmark position as (x,y) tuple.
        particles (list of Robot): The particles of the current step.
        m (int): The amount of particles.
        estimated_robot (Robot):
    """

    def __init__(self, robot, landmarks, m):
        """
        Init MCL object
        :param m: int, amount of particles
        :param landmarks: list, contains the landmarks positions
        :param robot: Robot, the robot we applied the MCL algorithm on it
        """

        self.landmarks = deepcopy(landmarks)
        self.robot_noise = deepcopy(robot.noise_std)
        self.particles = []
        self.estimated_robot = deepcopy(robot)
        self.m = m

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
        weights = []
        self.particles = []

        for i in tqdm(range(self.m)):

            # Create particle and move it
            particle = deepcopy(self.estimated_robot)
            particle.move(u_1, u_2)

            # Compute weight
            weight = self.compute_weight(measurements, particle)

            # Append particle and weight
            self.particles.append(particle)
            weights.append(weight)

        # Resample
        self.resample(weights)

        # Update estimated location
        self.update_estimation()

    def compute_weight(self, measurements, particle):

        # Sense the landmarks and compute probabilities
        particle_measurements = particle.sense(self.landmarks)
        prob = []

        # For each landmark measure
        for landmark_index, measure in enumerate(particle_measurements):
            landmark_prob = particle.measurement_probability(measurement=measurements[landmark_index],
                                                             landmark=self.landmarks[landmark_index])
            prob.append(landmark_prob)

        return np.prod(prob)

    def resample(self, weights, plot=True):

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

        estimated_position = np.mean([p.get_pose() for p in self.particles], axis=0)

        self.estimated_robot = Robot(init_pose=estimated_position, noise_std=self.robot_noise)
