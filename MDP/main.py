import pickle
from MDP.World import World
from MDP.mdp import TransitionModel, RewardFunction, MDP

# Action names
actions = ['north', 'east', 'south', 'west']

# Load Transition model
path = 'transition_model/{}.pkl'
transition_model = TransitionModel(actions = [a[0] for a in actions],
                                   transition_matrices=[pickle.load(open(path.format(a), 'rb')) for a in actions])

# Load reward function
path = 'reward_function/{}.pkl'
reward_matrices = [pickle.load(open(path.format(a), 'rb')) for a in actions]

# Create a MDP instance
mdp = MDP(World(), transition_model, reward_function, gamma=0.99)

# Solve it using value iteration algorithm
mdp.value_iteration(theta=10 ** -4, verbose=True)

# if __name__ == "__main__":
#     world = World()
#     world.plot()
#     world.plot_value([np.random.random() for i in range(world.nStates)])
#     world.plot_policy(np.random.randint(1, world.nActions, (world.nStates, 1)))
