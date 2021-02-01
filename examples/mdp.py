from MDP.mdp import TransitionModel, RewardFunction
import pickle


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
    actions = ['north', 'east', 'south', 'west']
    probabilities = load_matrices('transition_model')
    rewards = load_matrices('reward_function')

    # Transition model & reward function
    transition_model = TransitionModel(actions=[a[0] for a in actions], transition_matrices=probabilities)
    reward_function = RewardFunction(actions=[a[0] for a in actions], reward_matrices=rewards)


if __name__ == '__main__':
    main()
