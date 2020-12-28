from MCL.Robot import Robot
from MCL.World import World
import matplotlib.pyplot as plt
import numpy as np

# a.
# poses = [(45, 45, 0), (50, 60, np.pi / 2), (70, 30, 3 * np.pi / 4)]
# robots = [Robot(init_pose=p) for p in poses]
# world = World()
# world.plot(show=False)
# list(map(lambda x: x.plot(show=False), robots))
# plt.show()

# e., g.
robot_1 = Robot(init_pose=(10, 15, 0))
robot_2 = Robot(init_pose=(10, 15, 0), noise_std={'forward': 6, 'turn': 0.1, 'range': 5, 'bearing': 0.3})
moves = [(0, 60), (np.pi / 3, 30), (np.pi / 4, 30), (np.pi / 4, 20), (np.pi / 4, 40)]
world = World()
world.plot()
robot_1.plot()
robot_2.plot(mycolor='r')

for u_1, u_2 in moves:
    robot_1.move(u_1, u_2)
    robot_2.move(u_1, u_2)
    robot_1.plot()
    robot_2.plot(mycolor='r')

plt.plot(*(zip(*robot_1.path)), 'b:', label='Without noise')
plt.plot(*(zip(*robot_2.path)), 'r:', label='With noise')
plt.show()
