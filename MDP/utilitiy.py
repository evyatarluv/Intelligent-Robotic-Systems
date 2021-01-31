from typing import List, Dict, Union
import pandas as pd
import numpy as np


class TransitionModel:
    """
    The class represents a transition model for MDP algorithm.
    """
    def __init__(self, actions: List[str], transition_matrices: List[pd.DataFrame]):

        # Assert each action have is own matrix
        if len(actions) != len(transition_matrices):
            raise ValueError('The length of the actions must be equal to the length'
                             'of matrices')

        # Assert the sum of each row in all matrices equal to 1
        for mat in transition_matrices:
            if not all(mat.apply(np.sum, axis=1) == 1):
                raise ValueError('All rows in the matrices must sum to 1')

        self.actions: List[str] = [a.lower() for a in actions]
        self.states: List[int] = list(transition_matrices[0].columns)
        self.transition_model: Dict[int, pd.DataFrame] = {i: transition_matrices[i]
                                                          for i in range(len(actions))}

    def prob(self, target_state: int, current_state: int, action: Union[str, int]) -> float:
        """
        The method return the probability P(s' | s, a).
        :param target_state: s'
        :param action: a
        :param current_state: s
        :return: the desired probability
        """
        # Assert the target and current state are legal states
        if not all(s in self.states for s in [target_state, current_state]):
            raise ValueError('Not legal states.')

        # If the action was given as string convert it to the int
        if isinstance(action, str):
            action = self.actions.index(action.lower())

        return self.transition_model[action].at[current_state, target_state]
