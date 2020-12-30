from MCL.Robot import Robot
from MCL.World import World
from MCL.MCL import MCL

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------------------
# ----------------------------Example for using the MCL class --------------------------------------------
# --------------------------------------------------------------------------------------------------------
world = World()
robot = Robot(init_pose=(10, 15, 0), noise_std={'forward': 6, 'turn': 0.1, 'range': 5, 'bearing': 0.3})
moves = [(0, 60), (np.pi / 3, 30), (np.pi / 4, 30), (np.pi / 4, 20), (np.pi / 4, 40)]
mcl = MCL(robot, world.landmarks, 1000)

# Init plot
world.plot()
robot.plot()

for u in moves:

    robot.move(u[0], u[1])
    mcl.localize(u, robot.sense(world.landmarks))
    robot.plot()

# Plot paths
plt.plot(*(zip(*mcl.path)), '-.', color='limegreen', label='MCL')
plt.plot(*(zip(*robot.path)), '-.', color='royalblue', label='Robot')
plt.legend(bbox_to_anchor=(0.98, 1), loc='upper left')
plt.show()


