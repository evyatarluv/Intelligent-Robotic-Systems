from MCL.World import World
from MCL.Robot import Robot
from MCL.MCL import MCL
import numpy as np
from scipy.spatial import distance

# { n_particles: (mean, std)}
particles_result = {50: (4.594732281279549, 6.842995467244805),
                    250: (3.437128809250349, 2.214414191724139),
                    500: (3.3586272145613716, 1.8374038878898205),
                    1000: (3.460652485033129, 1.9068240876714986),
                    2000: (3.378818884555366, 1.849542994370264)}


def inspect_particles():
    """
    Inspection of the influence of particle's amount on the localization error.
    :return: list with each particle amount mean error and std
    """

    world = World()
    moves = [(0, 60), (np.pi / 3, 30), (np.pi / 4, 30), (np.pi / 4, 20), (np.pi / 4, 40)]
    result = {}
    n_repeat = 60
    n_particles = [50, 250, 500, 1000, 2000]

    for n in n_particles:

        print('Particles = {}'.format(n))
        errors = []

        for i in range(n_repeat):

            robot = Robot(init_pose=(10, 15, 0), noise_std={'forward': 6, 'turn': 0.1, 'range': 5, 'bearing': 0.3})
            mcl = MCL(robot, world.landmarks, n)

            for u in moves:
                # Move & estimate
                robot.move(u[0], u[1])
                mcl.localize(u, robot.sense(world.landmarks), plot=False)

                # Append the error
                errors.append(distance.euclidean(mcl.path[-1], robot.path[-1]))

        result[n] = (np.mean(errors), np.std(errors))

    print(result)
    return result
