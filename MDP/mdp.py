from typing import List, Union, Dict, Any
import pandas as pd
import numpy as np
from MDP.World import World


class TransitionModel:
    """
    The class represents a transition model for MDP algorithm.

    todo: add attributes description
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
    """
    todo: add class description

    todo: add attributes description
    """

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

    @staticmethod
    def change_transition_reward(rewards_matrix: pd.DataFrame,
                                 old_r: float, new_r: float) -> pd.DataFrame:
        """
        The method gets a reward function with a given reward for a transition and returns the matrix with a new reward
        for transition.
        :param rewards_matrix: the current reward matrix
        :param old_r: old transition's reward
        :param new_r: new transition reward
        :return: reward matrix after updating the transition's reward
        """

        rewards_matrix = rewards_matrix.applymap(lambda x: x if x == 0 else (x - old_r + new_r))

        return rewards_matrix


class MDP:
    """
    todo: add class description
    """

    def __init__(self, world: World, transition_model: TransitionModel, reward_function: RewardFunction,
                 gamma: float):

        self.world = world
        self.transition_model: TransitionModel = transition_model
        self.reward_function: RewardFunction = reward_function
        self.gamma: float = gamma
        self.states: List[int] = transition_model.states
        self.terminal_states: List[int] = world.stateTerminals
        self.actions: List[int] = list(range(world.nActions))
        self.value_function: Dict[int, float] = {i: 0 for i in self.states}
        self.policy: Dict[int, int] = {}

    def plot_values(self):
        """
        Plot the current values on a visual gridworld
        :return:
        """

        self.world.plot_value(self.value_function)

    def plot_policy(self):
        """
        Plot the current policy on the gridworld
        :return:
        """

        policy = np.zeros(self.world.nStates)

        for state, action in self.policy.items():

            policy[state - 1] = action + 1

        self.world.plot_policy(policy)

    def value_iteration(self, theta: float, verbose: bool = True):
        """
        The method implements the value iteration algorithm to find a policy.
        :param verbose: log prints during the algorithm
        :param theta: threshold for value improvement
        :return: policy as dict where the state is the key and the action is the value
        """

        values: Dict[int, float] = {i: 0 for i in self.states}
        iterations: int = 0

        # Verbose
        if verbose:
            print('Solving MDP using value iteration algorithm...')

        while True:

            delta = 0
            iterations += 1

            # Run through all the states
            for state in self.states:

                old_value = values[state]

                values[state] = self._optimal_value(state, values, 'value')

                delta = max(delta, np.abs(values[state] - old_value))

            # If the threshold was reached - break
            if delta < theta:
                break

        # Verbose
        if verbose:
            print('Convergence after {} iterations'.format(iterations))

        # Update value function & policy
        self.value_function = values
        self.policy = {s: self._optimal_value(s, self.value_function, 'action') for s in self.states}

    def _optimal_value(self, current_state: int, values: Dict[int, float], return_type: str) -> Any:
        """
        The method gets the current state and returns the optimal value-function, i.e., v*(s).
        :param return_type: determines if to return the value or action
        :param values: current value function
        :param current_state: s param in the equation.
        :return: optimal value or action.
        """

        action_values = [self._action_value_function(current_state, a, values) for a in self.actions]

        if return_type == 'action':
            return np.argmax(action_values)

        elif return_type == 'value':
            return max(action_values)

        else:
            raise NameError('Un-recognized return type, use `value` or `action` only')

    def _action_value_function(self, current_state: int, action: int, values: Dict[int, float]) -> float:
        """
        The method computed the action-value function for a given state (s), action (a) and current value function.
        :param current_state: param s in the equation
        :param action: param a in the equation
        :param values: param V in the equation
        :return: action value for the given input
        """
        R = 0
        value_function = 0

        for target_state in self.states:

            p = self.transition_model.prob(target_state, current_state, action)

            R += p * self.reward_function.reward(current_state, action, target_state)

            value_function += p * values[target_state]

        return R + self.gamma * value_function








