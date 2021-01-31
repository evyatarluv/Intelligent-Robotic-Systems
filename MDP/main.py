from MDP.World import World
from MDP.utilitiy import TransitionModel
import numpy as np
import os
import pickle

if __name__ == "__main__":

    # world = World()
    # world.plot()
    # world.plot_value([np.random.random() for i in range(world.nStates)])
    # world.plot_policy(np.random.randint(1, world.nActions, (world.nStates, 1)))

    # Test transition model
    matrices = []
    actions = ['n', 'e', 's', 'w']

    for a in ['north', 'east', 'south', 'west']:
        path = 'transition_model/{}.pkl'.format(a)
        matrices.append(pickle.load(open(path, 'rb')))

    tm = TransitionModel(actions, matrices)


