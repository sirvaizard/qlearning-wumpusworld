import random
import numpy as np

from wumpus import WumpusWorld


class Agent:

    def __init__(self):
        self.env = WumpusWorld()
        self.total_episodes = 5000
        self.max_step_per_episode = 100
        self.epsilon = 1.0              # Exploration rate %
        self.max_epsilon = 1.0          # Max exploration rate
        self.min_epsilon = 0.001        # Min exploration rate
        self.decay_rate = 0.01          # Exponential decay rate
        self.learning_rate = 0.1
        self.gamma = 0.99
        self.Qtable = np.zeros(
            (self.env.observation_space,
             self.env.action_space), dtype=np.float)

    def pick_action(self, state: int) -> int:
        if random.uniform(0, 1) > self.epsilon:
            # Take action with the highest Q-value (exploitation)
            action = np.argmax(self.Qtable[state, :])
        else:
            # Take a random action (exploration)
            action = np.random.randint(0, self.env.action_space)
        return action

    def train(self):
        max_reward = -np.inf
        for episode in range(self.total_episodes):
            state = self.env.reset()
            total_reward = 0

            # Reduce epsilon to get advantage of exploitation
            self.epsilon = (self.min_epsilon
                            + (self.max_epsilon - self.min_epsilon)
                            * np.exp(-self.decay_rate*episode))

            for step in range(self.max_step_per_episode):
                action = self.pick_action(state)
                if episode == self.total_episodes-1:
                    print(self.env.actions[action])
                    self.env.render()
                new_state, reward, done = self.env.step(action)
                total_reward += reward

                # Update the Q-table
                qvalue = self.Qtable[state, action]
                self.Qtable[state, action] = (
                    qvalue + self.learning_rate
                    * (reward + self.gamma
                       * np.max(self.Qtable[new_state]) - qvalue))

                state = new_state
                if done:
                    break
            if total_reward > max_reward:
                max_reward = total_reward
            if episode % 100 == 0:
                print(
                    f'episode: {episode} epsilon: {self.epsilon:.5f} ' +
                    f'reward: {total_reward} max-reward: {max_reward}')


if __name__ == '__main__':
    agent = Agent()
    agent.train()
