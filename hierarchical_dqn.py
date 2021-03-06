"""
Hierarchical DQN implementation as described in Kulkarni et al.
https://arxiv.org/pdf/1604.06057.pdf
@author: Saurabh Kumar
"""

from dqn import DqnAgent
import numpy as np
from qLearning import QLearningAgent


class HierarchicalDqnAgent(object):
    INTRINSIC_STEP_COST = -0.05

    def __init__(self,
                 learning_rates=[0.01, 0.01],
                 state_sizes=[0, 0],
                 agent_types=['network', 'network'],
                 subgoals=None,
                 num_subgoals=0,
                 num_primitive_actions=0,
                 meta_controller_state_fn=None,
                 check_subgoal_fn=None):
        """Initializes a hierarchical DQN agent.

           Args:
            learning_rates: learning rates of the meta-controller and controller agents.
            state_sizes: state sizes of the meta-controller and controller agents.
            agent_types: type of each agent - either tabular QLearning agent or Deep Q Network.
            subgoals: array of subgoals for the meta-controller.
            num_subgoals: the action space of the meta-controller.
            num_primitive_actions: the action space of the controller.
            meta_controller_state_fn: function that returns the state of the meta-controller.
            check_subgoal_fn: function that checks if agent has satisfied a particular subgoal.
        """

        self.agents = [None, None]
        num_actions_arr = [num_subgoals, num_primitive_actions]

        for i in xrange(len(agent_types)):
            if agent_types[i] == 'tabular':
                self.agents[i] = QLearningAgent(
                    num_states=state_sizes[i],
                    num_actions=num_actions_arr[i],
                    learning_rate=learning_rates[i],
                    epsilon=0.1)
            else:
                state_dims = []
                state_dims.append(state_sizes[i][0] + state_sizes[(i+1)%2])

                self.agents[i] = DqnAgent(
                    learning_rate=learning_rates[i],
                    num_actions=num_actions_arr[i],
                    state_dims=state_dims)

        self._meta_controller = self.agents[0]
        self._controller = self.agents[1]

        self._subgoals = subgoals
        self._num_subgoals = num_subgoals

        self._meta_controller_state_fn = meta_controller_state_fn
        self._check_subgoal_fn = check_subgoal_fn

        self._meta_controller_state = None
        self._curr_subgoal = None
        self._meta_controller_reward = 0

    def get_meta_controller_state(self, state):
        if not self._meta_controller_state_fn:
            return state
        else:
            # Augment the state with the reward.
            # s_with_reward = np.append(state[0], reward)
            # state_with_reward = np.array([s_with_reward])
            return self._meta_controller_state_fn(state)

    def get_controller_state(self, state, subgoal_index):
        curr_subgoal = self._subgoals[subgoal_index]
        # return [state.copy(), curr_subgoal.copy()]

        # Concatenate the environment state with the subgoal.
        controller_state = list(state[0])
        for i in xrange(len(curr_subgoal)):
            controller_state.append(i)
        controller_state = np.array([controller_state])
        return controller_state

    def intrinsic_reward(self, state, subgoal_index):
        if self.subgoal_completed(state, subgoal_index):
            return 1
        else:
            return self.INTRINSIC_STEP_COST

    def subgoal_completed(self, state, subgoal_index):
        if self._check_subgoal_fn is None:
            return state == self._subgoals[subgoal_index]
        else:

            # Augment the state with the reward.
            # s_with_reward = np.append(state[0], reward)
            # state_with_reward = np.array([s_with_reward])

            return self._check_subgoal_fn(state, subgoal_index)

    def store(self, state, action, reward, next_state, terminal, eval=False):
        """Stores the current transition in replay memory.
           The transition is stored in the replay memory of the controller.
           If the transition culminates in a subgoal's completion or a terminal state, a
           transition for the meta-controller is constructed and stored in its replay buffer.

           Args:
            state: current state
            action: primitive action taken
            reward: reward received from state-action pair
            next_state: next state
            terminal: extrinsic terminal (True or False)
            eval: Whether the current episode is a train or eval episode.
        """
        self._meta_controller_reward += reward

        # Compute the controller state, reward, next state, and terminal.
        intrinsic_state = self.get_controller_state(state, self._curr_subgoal)
        intrinsic_next_state = self.get_controller_state(next_state, self._curr_subgoal)
        intrinsic_reward = self.intrinsic_reward(next_state, self._curr_subgoal)
        intrinsic_terminal = self.subgoal_completed(
            next_state, self._curr_subgoal) or terminal


        self._controller.store(intrinsic_state, action,
            intrinsic_reward, intrinsic_next_state, intrinsic_terminal, eval)

        if self.subgoal_completed(next_state, self._curr_subgoal) or terminal:
            # if eval:
            #    if self.subgoal_completed(next_state, reward, self._curr_subgoal):
            #        print "Subgoal completed!"
            meta_controller_state = self._meta_controller_state.copy()
            next_meta_controller_state = self.get_meta_controller_state(next_state)
            self._meta_controller.store(meta_controller_state, self._curr_subgoal,
                self._meta_controller_reward, next_meta_controller_state, terminal, eval)

            # Reset the current meta-controller state and current subgoal to be None
            # since the current subgoal is finished. Also reset the meta-controller's reward.
            self._meta_controller_state = None
            self._curr_subgoal = None
            self._meta_controller_reward = 0

    def sample(self, state):
        """Samples an action from the hierarchical DQN agent.
           Samples a subgoal if necessary from the meta-controller and samples a primitive action
           from the controller.

           Args:
            state: the current environment state.

           Returns:
            action: a primitive action.
        """
        if self._meta_controller_state is None:
            self._meta_controller_state = self.get_meta_controller_state(state)
            self._curr_subgoal = self._meta_controller.sample(self._meta_controller_state)

        controller_state = self.get_controller_state(state, self._curr_subgoal)
        action = self._controller.sample(controller_state)

        return action

    def best_action(self, state):
        """Returns the greedy action from the hierarchical DQN agent.
           Gets the greedy subgoal if necessary from the meta-controller and gets
           the greedy primitive action from the controller.

           Args:
            state: the current environment state.

           Returns:
            action: the controller's greedy primitive action.
        """
        if self._meta_controller_state is None:
            self._meta_controller_state = self.get_meta_controller_state(state)
            self._curr_subgoal = self._meta_controller.best_action(self._meta_controller_state)
            # print "Current subgoal picked:"
            # print self._curr_subgoal

        controller_state = self.get_controller_state(state, self._curr_subgoal)
        action = self._controller.best_action(controller_state)

        # print "Primitive action:"
        # print action

        return action

    def update(self):
        self._controller.update()
        # Only update meta-controller right after a meta-controller transition has taken place,
        # which occurs only when either a subgoal has been completed or the agnent has reached a
        # terminal state.
        if self._meta_controller_state is None:
            self._meta_controller.update()
