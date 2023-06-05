import numpy as np
import pprint
import matplotlib.pyplot as plt
from Agent import QLearningAgent
from Environment import GridWorld

# Constants
NB_EPISODE = 20
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
ACTIONS = np.arange(4)

if __name__ == '__main__':
    grid_env = GridWorld()
    ini_state = grid_env.start_pos
    agent = QLearningAgent(
        alpha = ALPHA,
        gamma = GAMMA,
        epsilon = EPSILON,
        actions = ACTIONS,
        observation = ini_state,
    )
    rewards = []
    is_end_episode = False

    for episode in range(NB_EPISODE):
        episode_reward = []
        while not is_end_episode:
            action = agent.act()
            state, reward, is_end_episode = grid_env.step(action)
            agent.observe(state, reward)
            episode_reward.append(reward)
        rewards.append(np.sum(episode_reward))
        state = grid_env.reset()
        agent.observe(state)
        is_end_episode = False

    pprint.pprint(agent.q_values)
        
    plt.plot(np.arange(NB_EPISODE), rewards)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
