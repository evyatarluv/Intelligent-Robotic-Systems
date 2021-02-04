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
        The method implements the value iteration algorithm to find an optimal policy.
        :param verbose: log prints during the algorithm
        :param theta: threshold for value improvement
        :return:
        """

        value_function: Dict[int, float] = {i: 0 for i in self.states}
        iterations: int = 0

        # Verbose
        if verbose:
            print('Solving MDP using value iteration algorithm...')

        while True:

            delta = 0
            iterations += 1

            # Run through all the states
            for state in self.states:
                old_value = value_function[state]

                value_function[state] = self._optimal_value_function(state, value_function, 'value')

                delta = max(delta, np.abs(value_function[state] - old_value))

            # If the threshold was reached - break
            if delta < theta:
                break

        # Verbose
        if verbose:
            print('Convergence after {} iterations'.format(iterations))

        # Update value function & policy
        self.value_function = value_function
        self.policy = {s: self._optimal_value_function(s, self.value_function, 'action') for s in self.states}

    def _optimal_value_function(self, current_state: int, values: Dict[int, float], return_type: str) -> Any:
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

    def _action_value_function(self, current_state: int, action: int,
                               value_function: Dict[int, float]) -> float:
        """
        The method computed the action-value function for a given state (s), action (a) and current value function.
        The method returns q(s,a).
        :param current_state: param s in the equation
        :param action: param a in the equation
        :param value_function: param V in the equation
        :return: action value for the given input
        """

        action_value = 0

        for target_state in self.states:
            # Compute p(s'|s, a)
            p = self.transition_model.prob(target_state, current_state, action)

            # Compute r(s, a, s')
            r = self.reward_function.reward(current_state, action, target_state)

            # Update q(s, a)
            action_value += p * (r + self.gamma * value_function[target_state])

        return action_value

    def policy_iteration(self, theta: float, verbose: bool = True):
        """
        The method implements the policy iteration algorithm to find an optimal policy.
        :param verbose: if to print log messages
        :param theta: threshold for value function improvement
        :return:
        """

        # Init arbitrarily policy & value function
        policy = {s: np.random.randint(0, len(self.actions)) for s in self.states}
        value_function = {s: 0 for s in self.states}
        iterations = 0

        # Verbose
        if verbose:
            print('Solving MDP using policy iteration algorithm...')

        # Policy iteration
        while True:

            iterations += 1

            # plot
            self.policy = policy
            self.value_function = value_function
            self.plot_policy()
            self.plot_values()

            # Policy evaluation
            new_value_function = self._policy_evaluation(policy, theta)

            # Policy Improvement
            new_policy = self._policy_improvement(value_function)

            # Break if policy & value function converge
            if self._converge(policy, new_policy, value_function, new_value_function, theta):
                break

            else:
                policy = new_policy
                value_function = new_value_function

        # Verbose
        if verbose:
            print('Convergence after {} iterations'.format(iterations))

        self.policy = new_policy
        self.value_function = new_value_function

    def _policy_evaluation(self, policy: Dict[int, int], theta: float) -> Dict[int, int]:
        """
        The method estimate the value function for a given policy.
        Implements the iterative policy evaluation.
        :param policy: dict which represent the current policy
        :param theta: threshold for breaking the policy evaluation loop
        :return: value function v(s)
        """

        # Init arbitrarily value function
        value_function = {s: 0 for s in self.states}

        # Evaluate policy until converge
        while True:

            delta = 0

            for s in self.states:

                old_value = value_function[s]

                value_function[s] = self._action_value_function(s, policy[s], value_function)

                delta = max(delta, np.abs(value_function[s] - old_value))

            # If the threshold was reached - break
            if delta < theta:
                break

        return value_function

    def _policy_improvement(self, value_function: Dict[int, float]) -> Dict[int, int]:
        """
        The method gets a value function, v(s), and use it to improve the policy
        following a greedy strategy.
        :param value_function: value for each state.
        :return: dict with action for each state
        """
        # Init policy
        policy = {}

        # Improve policy according given value function
        for s in self.states:

            # Get q(s,a) for each a in actions
            actions_values = [self._action_value_function(s, a, value_function) for a in self.actions]

            # Update policy for state s
            policy[s] = int(np.argmax(actions_values))

        return policy

    def _converge(self, policy: Dict[int, int], new_policy: Dict[int, int],
                  value_function: Dict[int, float], new_value_function: Dict[int, float], theta: float) -> bool:

        # Assert all states in policy & value function
        assert len(policy) == len(new_policy) == len(self.states), 'Missing states in policies'
        assert len(value_function) == len(new_value_function) == len(self.states), 'Missing states in value functions'

        # Look for changes in policy
        for s in policy.keys():

            if policy[s] != new_policy[s]:

                return False

        # Look for changes in value function
        for s in value_function.keys():

            if np.abs(value_function[s] - new_value_function[s]) > theta:

                return False

        # If you pass all checks return True
        return True




