from MCL.Robot import Robot
from MCL.World import World
import matplotlib.pyplot as plt
import numpy as np

poses = [(45, 45, 0), (50, 60, np.pi / 2), (70, 30, 3 * np.pi / 4)]
robots = [Robot(init_pose=p) for p in poses]
world = World()
