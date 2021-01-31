from MDP.utilitiy import reward_function, TransitionModel
import pickle


def load_matrices():
    """
    Load the probability matrix for each action.
    :return:
    """
    actions = ['north', 'east', 'south', 'west']
    matrices = []

    for a in actions:
        path = '../MDP/transition_model/{}.pkl'.format(a)
        matrices.append(pickle.load(open(path, 'rb')))
        # print('\nAction = {} \n{}'.format(a.capitalize(), matrices[-1]))

    return matrices


def main():
    actions = ['north', 'east', 'south', 'west']
    prob_matrices = load_matrices()

    # Transition model & reward function
    transition_model = TransitionModel(actions=[a[0] for a in actions],
                                       transition_matrices=prob_matrices)

    rewards = reward_function(holes=[1, 7, 14], goals=[13], obstacles=[15])


if __name__ == '__main__':
    main()
