class Agent:
    def __init__(self, action_space, *args):
        self.NUM_ACTIONS = action_space.n

    def load(self, *args):
        pass

    def save(self, *args):
        pass

    def learn(self, *args):
        pass
    
    def clean_up(self, *args):
        pass

    def pick_action(self, *args):
        raise NotImplementedError
