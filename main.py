from MCL.Robot import Robot
from MCL.World import World
from MCL.MCL import MCL

import matplotlib.pyplot as plt
import numpy as np

r = Robot(init_pose=(10, 15, 0), noise_std={'forward': 6, 'turn': 0.1, 'range': 5, 'bearing': 0.3})
w = World()
mcl = MCL(r, w.landmarks, 1000)
w.plot()
r.plot()
u = (0, 60)
r.move(u[0], u[1])
mcl.localize(u, r.sense(w.landmarks))




for p in mcl.particles:
    p.plot(style='particle', mycolor='black')

r.plot()
plt.show()

