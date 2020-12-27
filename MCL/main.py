from MCL.Robot import Robot
from MCL.World import World
import matplotlib.pyplot as plt
import numpy as np

# a.
poses = [(45, 45, 0), (50, 60, np.pi / 2), (70, 30, 3 * np.pi / 4)]
robots = [Robot(init_pose=p) for p in poses]
world = World()
world.plot(show=False)
list(map(lambda x: x.plot(show=False), robots))
plt.show()

# b.
