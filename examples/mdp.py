from MDP.utilitiy import reward_function


def main():
    rewards = reward_function(holes=[1, 7, 14],
                              goals=[13],
                              obstacles=[15])

if __name__ == '__main__':

    main()