from MCL.Robot import Robot
from MCL.World import World
from MCL.MCL import MCL

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------------------------
# Init both robots
world = World()
moves = [(0, 60), (np.pi / 3, 30), (np.pi / 4, 30), (np.pi / 4, 20), (np.pi / 4, 40)]
robot_1 = Robot(init_pose=(10, 15, 0))
robot_2 = Robot(init_pose=(10, 15, 0), noise_std={'forward': 6, 'turn': 0.1, 'range': 5, 'bearing': 0.3})

# Init plot
world.plot()
robot_1.plot()
robot_2.plot(mycolor='r')

# Move both robots and plot the moves
for u_1, u_2 in moves:
    robot_1.move(u_1, u_2)
    robot_2.move(u_1, u_2)
    robot_1.plot()
    robot_2.plot(mycolor='r')

# Add the robot path
plt.plot(*(zip(*robot_1.path)), 'b:', label='Without noise')
plt.plot(*(zip(*robot_2.path)), 'r:', label='With noise')
plt.legend(bbox_to_anchor=(0.98, 1), loc='upper left')
plt.show()

# --------------------------------------------------------------------------------------------------------
# r = Robot(init_pose=(10, 15, 0), noise_std={'forward': 6, 'turn': 0.1, 'range': 5, 'bearing': 0.3})
# w = World()
# moves = [(0, 60), (np.pi / 3, 30), (np.pi / 4, 30), (np.pi / 4, 20), (np.pi / 4, 40)]
#
# mcl = MCL(r, w.landmarks, 1000)
# w.plot()
# r.plot()
#
# for u in moves:
#
#     r.move(u[0], u[1])
#     sample, resample = mcl.localize(u, r.sense(w.landmarks))
#
#     for i in range(len(sample)):
#         sample[i].plot(style='particle', mycolor='black', markersize=2)
#         resample[i].plot(style='particle', mycolor='lightgrey', markersize=2)
#
#     r.plot()
#
# plt.show()

