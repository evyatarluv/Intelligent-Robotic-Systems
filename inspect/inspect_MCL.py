from scipy import stats
from tqdm import tqdm

from MCL.World import World
from MCL.Robot import Robot
from MCL.MCL import MCL
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def inspect_particles(n_repeat, n_particles):
    """
    Inspection of the influence of particle's amount on the localization error.
    :return: result as a dict
    """

    world = World()
    moves = [(0, 60), (np.pi / 3, 30), (np.pi / 4, 30), (np.pi / 4, 20), (np.pi / 4, 40)]
    result = {}

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

        print('Mean Error: {}'.format(np.mean(errors)))
        result[n] = errors

    pd.DataFrame.from_dict(result).to_csv('particles_results.csv', index=False)
    return result


# inspect_particles(n_repeat=60, n_particles=[50, 250, 500, 1000, 2000])

def plot_inspect_particles(results_path):

    # Plot results
    df = pd.read_csv(results_path)
    n_particles = [50, 250, 500, 1000, 2000]
    means = df.apply(np.mean)
    stds = df.apply(np.std)
    plt.errorbar(range(len(means)), means, yerr=stds / 4, capsize=2.5, fmt='-o')
    plt.ylim((0, 7))
    plt.xlabel('Amount of Particles')
    plt.ylabel('MCL Error')
    plt.title('Influence of Particles Amount on MCL Error')
    plt.xticks(range(len(means)), [str(p) for p in n_particles])
    plt.grid(True, axis='y')
    plt.show()


plot_inspect_particles('particles_results.csv')