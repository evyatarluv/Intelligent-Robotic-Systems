from typing import List, Union, Dict
import pandas as pd
import numpy as np


class TransitionModel:
    """
    The class represents a transition model for MDP algorithm.
    """
    def __init__(self, actions: List[str], transition_matrices: List[pd.DataFrame]):

        # Assert each action have is own matrix
        assert len(actions) == len(transition_matrices), 'Actions length must be equal to matrices length'

        # Assert the sum of each row in all matrices equal to 1
        for mat in transition_matrices:
            assert all(mat.apply(np.sum, axis=1) == 1), 'All rows in a probability matrix must sum to 1'

        self.actions: List[str] = [a.lower() for a in actions]
        self.states: List[int] = list(transition_matrices[0].columns)
        self.transition_model: Dict[int, pd.DataFrame] = {i: transition_matrices[i]
                                                          for i in range(len(actions))}

    def prob(self, target_state: int, current_state: int, action: int) -> float:
        """
        The method return the probability P(s' | s, a).
        :param target_state: s'
        :param action: a
        :param current_state: s
        :return: the desired probability
        """
        # Assert the target and current state are legal states
        assert all(s in self.states for s in [target_state, current_state]), 'Not legal states'

        # Assert the action enum is legal
        assert action < len(self.actions), 'Not legal action'

        return self.transition_model[action].at[current_state, target_state]


class RewardFunction:

    def __init__(self, actions: List[str], reward_matrices: List[pd.DataFrame]):

        # Assert each action have is own matrix
        if len(actions) != len(reward_matrices):
            raise ValueError('The length of the actions must be equal to the length'
                             'of matrices')

        self.actions: List[str] = [a.lower() for a in actions]
        self.states: List[int] = list(reward_matrices[0].columns)
        self.reward_function: Dict[int, pd.DataFrame] = {i: reward_matrices[i]
                                                         for i in range(len(actions))}

    def reward(self, s: int, a: int, s_prime: int) -> float:
        """
        The method returns the reward of r(s, a, s').
        :param s: state in time t-1
        :param a: action in time t-1
        :param s_prime: state in time t
        :return: reward in time t
        """
        # Assert s & s_prime are legal states
        assert all(s in self.states for s in [s, s_prime]), 'Not legal states'

        # Assert the action enum is legal
        assert a < len(self.actions), 'Not legal action'

        return self.reward_function[a].at[s, s_prime]


class MDP:

    pass



