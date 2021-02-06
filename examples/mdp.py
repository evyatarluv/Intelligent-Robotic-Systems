from MDP.mdp import TransitionModel, RewardFunction, MDP
from MDP.World import World
import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_matrices(directory: str):
    """
    Load the probability/reward matrix for each action.
    :return:
    """
    actions = ['north', 'east', 'south', 'west']
    matrices = []

    for a in actions:
        path = '../MDP/{}/{}.pkl'.format(directory, a)
        matrices.append(pickle.load(open(path, 'rb')))
        # print('\nAction = {} \n{}'.format(a.capitalize(), matrices[-1]))

    return matrices


def main():

    # load MDP example data
    actions = ['north', 'east', 'south', 'west']
    probabilities = load_matrices('transition_model')
    rewards = load_matrices('reward_function')

    # Transition model & reward function
    transition_model = TransitionModel(actions=[a[0] for a in actions], transition_matrices=probabilities)
    reward_function = RewardFunction(actions=[a[0] for a in actions], reward_matrices=rewards)

    # Find optimal policy using different algorithm and parameters
    # Value iteration algorithm
    mdp = MDP(World(), transition_model, reward_function, gamma=0.99)
    mdp.value_iteration(theta=10 ** -4)
    mdp.plot_values()
    mdp.plot_policy()

    # Policy iteration
    mdp = MDP(World(), transition_model, reward_function, gamma=0.9)
    mdp.policy_iteration(theta=10 ** -4)
    mdp.plot_values()
    mdp.plot_policy()


if __name__ == '__main__':
    main()
