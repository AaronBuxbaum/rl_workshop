from collections import deque
from random import sample
import numpy as np
import gym
import keras
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from agent import Agent


EPSILON_MIN = 0.1
EPSILON_MAX = 0.8
EPSILON_DECAY = 0.00075
MEMORY_CAPACITY = 500000
TARGET_UPDATE = 300
SIZE_HIDDEN = 16
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.0075
MAX_STEPS = 2000
ACTIVATION = 'tanh'
LEARNING_START = 100
N_EPISODES = 20000
MONITOR_DIR = '/tmp/cartpole4'

class DQNAgent(Agent):
    def __init__(self, action_space, observation_space):
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.dim_actions = action_space.n
        self.dim_states = observation_space.shape
        self.NN = NN(observation_space.shape, action_space.n,
                     BATCH_SIZE, SIZE_HIDDEN, LEARNING_RATE, ACTIVATION)
        self.observers = []
        self.episode_count = 0
        self.step_count_total = 1
        self.step_count_episode = 1
        self.epsilon_min = EPSILON_MIN
        self.epsilon_max = EPSILON_MAX
        self.epsilon_decay = EPSILON_DECAY
        self.target_update = TARGET_UPDATE
        self.max_steps = MAX_STEPS
        self.n_episodes = N_EPISODES
        self.epsilon = EPSILON_MAX
        self.batch_size = BATCH_SIZE
        self.usetarget = False
        self.gamma = GAMMA
        self.loss = 0
        self.done = False
        self.reward = 0
        self.reward_episode = 0
        self.learning_switch = False
        self.learning_start = LEARNING_START

    def notify(self, event):
        for observer in self.observers:
            observer(event)
        pass

    def pick_action(self, state):
        self.step_count_total += 1
        action = self.choose_action(state)
        return action

    def learn(self, state, new_state, action, reward, done):
        obs = (state, action, new_state, reward, done)
        self.memory.store(obs)
        if self.learning_switch:
            self.backup()
        self.notify('step_done')
        pass

    def backup(self):
        self.flashback()
        if self.step_count_total % self.target_update == 0:
            self.NN.update_target()
            self.usetarget = True
        pass

    def flashback(self):
        X, y = self._make_batch()
        self.loss = self.NN.train(X, y)
        # if np.isnan(self.loss.history['loss']).any():
        #     print('Warning, loss is {}'.format(self.loss))
        pass

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            choice = self.random_choice()
        else:
            choice = self.greedy_choice(state)
        return choice

    def greedy_choice(self, state):
        greedy_choice = self.NN.best_action(state, usetarget=False)
        return greedy_choice

    def random_choice(self):
        random_choice = np.random.randint(0, self.dim_actions)
        return random_choice

    def _make_batch(self):
        X = []
        y = []
        batch = self.memory.get_batch(self.batch_size)
        for state, action, newstate, reward, done in batch:
            X.append(state)
            target = self.NN.predict(state, False)
            q_vals_new_t = self.NN.predict(newstate, self.usetarget)
            a_select = self.NN.best_action(newstate, False)
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * q_vals_new_t[a_select]
            y.append(target)
        return X, y

    def add_observer(self, observer):
        self.observers.append(observer)
        pass


class NN:
    def __init__(self, n_states, n_actions, batch_size, size_hidden,
                 learning_rate, activation):
        self.learning_rate = learning_rate
        self.act = activation
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = self._make_model(n_states, n_actions, size_hidden)
        self.model_t = self._make_model(n_states, n_actions, size_hidden)
        self.batch_size = batch_size

    def _make_model(self, n_states, n_actions, size_hidden):
        model = Sequential()
        model.add(Dense(size_hidden, input_dim=4, activation=self.act))
        model.add(Dense(size_hidden, activation=self.act))
        model.add(Dense(n_actions, activation='linear'))
        opt = SGD(lr=self.learning_rate, momentum=0.5, decay=1e-6, clipnorm=2)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    def train(self, X, y):
        X = prep_batch(X)
        y = prep_batch(y)
        loss = self.model.fit(X,
                              y,
                              batch_size=self.batch_size,
                              epochs=1,
                              verbose=0,
                              shuffle=True)

        return loss

    def predict(self, state, usetarget=False):
        state = prep_input(state, self.n_states[0])
        if usetarget:
            q_vals = self.model_t.predict(state)
        else:
            q_vals = self.model.predict(state)
        return q_vals[0]

    def update_target(self):
        weights = self.model.get_weights()
        self.model_t.set_weights(weights)
        self.save('weights.h5')
        pass

    def best_action(self, state, usetarget=False):
        state = prep_input(state, self.n_states[0])
        q_vals = self.predict(state, usetarget)
        best_action = np.argmax(q_vals)
        return best_action

    def save(self, fname):
        self.model.save_weights(fname, overwrite=True)
        pass

    def load(self, fname):
        self.model.load_weights(fname)
        self.update()
        pass


class ReplayMemory:
    def __init__(self, capacity):
        self.samples = deque([], maxlen=capacity)

    def store(self, exp):
        self.samples.append(exp)
        pass

    def get_batch(self, n):
        n_samples = min(n, len(self.samples))
        samples = sample(self.samples, n_samples)
        return samples


class EpsilonUpdater:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, event):
        if event == 'step_done':
            self.epsilon_update()
            self.switch_learning()
        else:
            pass

    def epsilon_update(self):
        self.agent.epsilon = (
            self.agent.epsilon_min +
            (self.agent.epsilon_max - self.agent.epsilon_min) * np.exp(
                -self.agent.epsilon_decay * self.agent.step_count_total))
        pass

    def switch_learning(self):
        if self.agent.step_count_total >= self.agent.learning_start:
            self.agent.learning_switch = True
        pass


def prep_input(data, n_dimension):
    prep = np.asarray(data)
    transformed = prep.reshape((1, n_dimension))
    return transformed


def prep_batch(to_prep):
    prep = np.vstack(to_prep)
    return prep_batch
