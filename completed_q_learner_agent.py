import math
import numpy as np
from agent import Agent

class CompletedQLearnerAgent(Agent):
    def __init__(self, action_space, observation_space):
        super(CompletedQLearnerAgent, self).__init__(action_space, observation_space)
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

    # Learn with Q-Learning
    def learn(self, state, new_state, action, reward):
        state = self.discretize(state)
        new_state = self.discretize(new_state)
        old_value = self.Q[state][action]
        known_value = (1 - self.learning_rate) * old_value
        future_value = np.max(self.Q[new_state])
        learned_value = self.learning_rate * (reward + self.discount_rate * future_value)
        self.Q[state][action] = known_value + learned_value

    # Pick the right action for the moment
    def pick_action(self, state):
        if np.random.random() < self.temperature:
            return np.random.randint(0, self.NUM_ACTIONS)
        return self.pick_best_action(state)

    # Pick the best action, as far as we know
    def pick_best_action(self, state):
        state = self.discretize(state)
        return np.argmax(self.Q[state])

    # Decay temperature and learning rate
    def clean_up(self, iteration):
        self.temperature = self.decay_with_iteration(iteration)
        self.learning_rate = self.decay_with_iteration(iteration)
