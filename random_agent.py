import numpy as np
from agent import Agent

class RandomAgent(Agent):
    def pick_action(self, state):
        return np.random.randint(0, self.NUM_ACTIONS)
