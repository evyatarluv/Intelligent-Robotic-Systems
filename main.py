from MCL.Robot import Robot
from MCL.World import World
from MCL.MCL import MCL

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------------------------
# ----------------------------Example for using the MCL class --------------------------------------------
# --------------------------------------------------------------------------------------------------------
world = World()
robot = Robot(init_pose=(10, 15, 0),
              noise_std={'forward': 6, 'turn': 0.1, 'range': 5, 'bearing': 0.3})
moves = [(0, 60), (np.pi / 3, 30), (np.pi / 4, 30), (np.pi / 4, 20), (np.pi / 4, 40)]
n_particles = 1000
mcl = MCL(robot, world.landmarks, n_particles)

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


