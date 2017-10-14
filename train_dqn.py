"""
@author: Saurabh Kumar
"""

import os
import clustering
import dqn
import gym
from gym.wrappers import Monitor
import hierarchical_dqn
import numpy as np


def log(logfile, iteration, rewards):
    """Function that logs the reward statistics obtained by the agent.

    Args:
        logfile: File to log reward statistics.
        iteration: The current iteration.
        rewards: Array of rewards obtained in the current iteration.
    """
    log_string = '{} {} {} {}'.format(
        iteration, np.min(rewards), np.mean(rewards), np.max(rewards))
    print(log_string)

    with open(logfile, 'a') as f:
        f.write(log_string + '\n')


def make_environment(env_name):
    return gym.make(env_name)


def make_agent(agent_type, env, use_clustering):
    if agent_type == 'dqn':
        return dqn.DqnAgent(state_dims=[2],
                            num_actions=env.action_space.n)
    elif agent_type == 'h_dqn':
        meta_controller_state_fn, check_subgoal_fn, num_subgoals, subgoals = clustering.get_cluster_fn()

        return hierarchical_dqn.HierarchicalDqnAgent(
            state_sizes=[num_subgoals, [2]],
            agent_types=['tabular', 'network'],
            subgoals=subgoals,
            num_subgoals=num_subgoals,
            num_primitive_actions=env.action_space.n,
            meta_controller_state_fn=meta_controller_state_fn,
            check_subgoal_fn=check_subgoal_fn)


def run(env_name='MountainCar-v0',
        agent_type='dqn',
        num_iterations=100,
        num_train_episodes=1000,
        num_eval_episodes=100,
        use_clustering=False,
        logfile=None):
    """Function that executes RL training and evaluation.

    Args:
        env_name: Name of the environment that the agent will interact with.
        agent_type: The type RL agent that will be used for training.
        num_iterations: Number of iterations to train for.
        num_train_episodes: Number of training episodes per iteration.
        num_eval_episodes: Number of evaluation episodes per iteration.
        use_clustering: Whether or not to perform unsupervised clustering - applicable for h-DQN.
        logfile: File to log the agent's performance over training.
    """

    env = make_environment(env_name)
    env_test = make_environment(env_name)
    # env_test = Monitor(env_test, directory='videos/', video_callable=lambda x: True, resume=True)
    print 'Made environment!'
    agent = make_agent(agent_type, env, use_clustering)
    print 'Made agent!'

    for it in range(num_iterations):


        # Run train episodes.
        for train_episode in range(num_train_episodes):
            # Reset the environment.
            state = env.reset()
            state = np.expand_dims(state, axis=0)

            episode_reward = 0

            # Run the episode.
            terminal = False

            while not terminal:
                action = agent.sample(state)

                next_state, reward, terminal, _ = env.step(action)
                next_state = np.expand_dims(next_state, axis=0)

                agent.store(state, action, reward, next_state, terminal)
                agent.update()

                episode_reward += reward
                # Update the state.
                state = next_state

        eval_rewards = []

        # Run eval episodes.
        for eval_episode in range(num_eval_episodes):

            # Reset the environment.
            state = env_test.reset()
            state = np.expand_dims(state, axis=0)

            episode_reward = 0

            # Run the episode.
            terminal = False

            while not terminal:
                action = agent.best_action(state)

                next_state, reward, terminal, _ = env_test.step(action)
                next_state = np.expand_dims(next_state, axis=0)

                agent.store(state, action, reward, next_state, terminal, eval=True)

                episode_reward += reward

                state = next_state

            eval_rewards.append(episode_reward)

        log(logfile, it, eval_rewards)


logfile = 'experiment_logs/exp_hrl_100_clustering_episodes.txt'
env_name = 'MountainCar-v0'
agent_type ='h_dqn'
run(env_name=env_name, agent_type=agent_type, logfile=logfile, use_clustering=True)