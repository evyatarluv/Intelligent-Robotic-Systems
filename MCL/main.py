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

# e.
robot = Robot(init_pose=(10, 15, 0))
moves = [(0, 60), (np.pi / 3, 30), (np.pi / 4, 30), (np.pi / 4, 20), (np.pi / 4, 40)]
world = World()
world.plot()
robot.plot()

for m in moves:
    robot.move(m[0], m[1])
    robot.plot()

plt.plot(*(zip(*robot.path)), ':')
plt.show()
