import argparse
import sys
import gym
import numpy as np
import os
from gym import wrappers, logger
from random_agent import RandomAgent
from q_learner_agent import QLearnerAgent
from completed_q_learner_agent import CompletedQLearnerAgent
from double_deep_q_learner_agent import DDQNAgent, EpsilonUpdater

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--environment', nargs='?', default='CartPole-v0', help='Select the environment to run')
    parser.add_argument('--agent', nargs='?', default='QLearnerAgent', help='Select an agent')
    parser.add_argument('--monitor', nargs='?', default=True, help='Decide if you want monitor files')
    parser.add_argument('--episodes', nargs='?', default=1024, help='Select number of episodes to run')
    parser.add_argument('--load', nargs='?', default=False, help='Decide if you want to load previous information')
    parser.add_argument('--maxsteps', nargs='?', default=1000, help='Select number of time steps for CartPole before maximum is reached') # make the game much harder

    args = parser.parse_args()
    logger.set_level(logger.INFO)
    env = gym.make(args.environment)
    if args.environment == 'CartPole-v0':
        env._max_episode_steps = int(args.maxsteps)
    if args.monitor:
        outdir = os.path.join(os.getcwd(), "recordings")
        env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(420)
    np.random.seed(420)

    agent = eval(args.agent)(env.action_space, env.observation_space) # NEVER USE EVAL IN REAL LIFE!!!
    if args.agent == 'DDQNAgent':
        epsilon = EpsilonUpdater(agent)
        agent.add_observer(epsilon)

    if args.load:
        agent.load()

    for i in range(int(args.episodes)):
        state = env.reset()
        while True:
            action = agent.pick_action(state)
            newState, reward, done, _ = env.step(action)
            agent.learn(state, newState, action, reward, done)
            agent.clean_up(i)
            state = newState
            if done:
                break

    agent.save()
    env.env.close()
