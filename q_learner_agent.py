import math
import numpy as np
from agent import Agent

class QLearnerAgent(Agent):
    def __init__(self, action_space, observation_space):
        super(QLearnerAgent, self).__init__(action_space, observation_space)
        self.Q_TABLE_FILE = 'q_table.npy'

        # Q-Learning variables
        self.learning_rate = 1.0
        self.discount_rate = 1.0
        self.temperature = 1.0

        # Specific to this problem for sake of simplicity
        self.buckets = (1, 1, 6, 12)
        self.Q = np.zeros(self.buckets + (action_space.n,))
        self.upper_bounds = [observation_space.high[0], 0.5, observation_space.high[2], math.radians(50)]
        self.lower_bounds = [observation_space.low[0], -0.5, observation_space.low[2], -math.radians(50)]
    
    # Load a Q-table file
    def load(self):
        try:
            Q = np.load(self.Q_TABLE_FILE)
            if Q.shape == self.Q.shape:
                self.Q = Q
        except:
            pass

    # Save a Q-table file
    def save(self):
        np.save(self.Q_TABLE_FILE, self.Q)

    # Convert continuous observation state to a tuple
    def discretize(self, state):
        ratios = [(state[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i]) for i in range(len(state))]
        new_state = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(state))]
        new_state = [min(self.buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
        return tuple(new_state)

    # Decay vs iterations, including min and max values
    def decay_with_iteration(self, iteration):
        return max(0.1, min(1.0, 1.0 - math.log10((iteration + 1) / 25)))

    # TODO
    def learn(self, state, new_state, action, reward, done):
        pass

    # TODO
    def pick_action(self, state):
        pass

    # TODO
    def pick_best_action(self, state):
        pass

    # TODO
    def clean_up(self, iteration):
        pass
